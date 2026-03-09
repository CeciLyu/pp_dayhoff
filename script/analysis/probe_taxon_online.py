"""
Train a hierarchical taxonomic linear probe with on-the-fly hidden extraction.

Instead of pre-extracting and storing hidden states to disk, this script:
  1. Loads the frozen pretrained model once.
  2. For each batch of orthologous groups, extracts hidden states on-the-fly.
  3. Trains a single species-level linear probe with a hierarchical taxonomic
     loss that aggregates species probabilities up through genus, family, order,
     class, phylum, and domain.

This eliminates the large storage requirement and the I/O bottleneck of
reading pre-extracted hidden states from disk.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import random
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from deimm.model.tokenizer import PureARTokenizer
from deimm.utils.constants import MSA_PAD, PRETRAIN_DIR, RANK
from deimm.utils.training_utils import load_convert_parent, seed_everything

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_TAXONOMY_MAPPING = "/scratch/suyuelyu/deimm/data/oma/taxid_to_std_ranks.pkl"
DEFAULT_TRAIN_PARQUET_GLOB = (
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_train_chunk_*.parquet"
)
DEFAULT_VAL_PARQUET = (
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_val.parquet"
)
DEFAULT_TEST_PARQUET = (
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet"
)
DEFAULT_OUTPUT_DIR = "/scratch/suyuelyu/deimm/results/probe_taxon/online_hierarchical"

TAXONOMY_RANKS = ["genus", "family", "order", "class", "phylum", "domain"]
TOKENIZER_CONFIG = {"vocab_path": "vocab_UL_ALPHABET_PLUS.txt", "allow_unk": False}

HIDDEN_DIM = 1280


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a hierarchical taxonomic probe with on-the-fly hidden extraction."
    )

    # Data
    p.add_argument("--train_parquet_glob", type=str, default=DEFAULT_TRAIN_PARQUET_GLOB)
    p.add_argument("--val_parquet", type=str, default=DEFAULT_VAL_PARQUET)
    p.add_argument("--test_parquet", type=str, default=DEFAULT_TEST_PARQUET)
    p.add_argument(
        "--taxonomy_mapping_file", type=str, default=DEFAULT_TAXONOMY_MAPPING
    )
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # Model
    p.add_argument(
        "--layer_idx",
        type=int,
        default=-1,
        help="Which hidden layer to use for the probe (-1 = last).",
    )
    p.add_argument(
        "--n_max_protein",
        type=int,
        default=64,
        help="Max proteins per OG to feed through the pretrained model.",
    )

    # Training
    p.add_argument(
        "--n_sample_per_og_train",
        type=int,
        default=1,
        help="Number of samples to take per OG per epoch during training (for speed).",
    )
    p.add_argument(
        "--n_sample_per_og_eval",
        type=int,
        default=1,
        help="Number of samples to take per OG during evaluation (should be higher).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--min_class_count", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--accumulation_steps",
        type=int,
        default=16,
        help="Number of OG forward passes before one optimizer step.",
    )
    p.add_argument(
        "--amp", action="store_true", help="Enable mixed precision for probe training."
    )

    # Hierarchical loss
    p.add_argument(
        "--rank_weights",
        type=str,
        default="species:1.0,genus:0.5,family:0.3,order:0.2,class:0.1,phylum:0.05,domain:0.02",
        help="Comma-separated rank:weight pairs for the hierarchical loss.",
    )
    p.add_argument(
        "--class_weight_mode",
        type=str,
        default="balanced",
        choices=["balanced", "none", "log"],
    )

    # Scheduler
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["none", "reduce_on_plateau"],
    )
    p.add_argument("--lr_scheduler_patience", type=int, default=2)
    p.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    p.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)

    # Eval & checkpointing
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--resume_from", type=str, default=None)

    return p.parse_args()


def parse_rank_weights(s: str) -> dict[str, float]:
    weights = {}
    for pair in s.split(","):
        rank, w = pair.strip().split(":")
        weights[rank.strip()] = float(w.strip())
    return weights


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    shutil.copy2(Path(__file__).resolve(), output_dir / Path(__file__).name)


# ──────────────────────────────────────────────────────────────────────
# Taxonomy helpers
# ──────────────────────────────────────────────────────────────────────
def load_full_taxonomy(
    mapping_file: str,
) -> dict[int, dict[str, int]]:
    """Load taxid_to_std_ranks: {species_taxid: {rank: taxid, ...}}."""
    with open(mapping_file, "rb") as f:
        return pickle.load(f)


def build_species_classes(
    taxid_to_std_ranks: dict[int, dict[str, int]],
    min_class_count: int,
    train_df: pd.DataFrame,
) -> tuple[LabelEncoder, dict[int, int], set[int]]:
    """
    Count species in training data, filter by min_class_count, and build
    a LabelEncoder + species_tid -> class_idx mapping.
    """
    species_counts: Counter = Counter()
    for _, row in train_df.iterrows():
        for tid in row["taxid"]:
            tid = int(tid)
            if tid in taxid_to_std_ranks:
                species_counts[tid] += 1

    valid_species = {s for s, n in species_counts.items() if n >= min_class_count}
    if len(valid_species) < 2:
        raise RuntimeError(
            f"Need >= 2 species after filtering (got {len(valid_species)}). "
            "Lower --min_class_count."
        )

    le = LabelEncoder()
    le.fit(sorted(valid_species))
    tid_to_cls = {int(s): int(le.transform([s])[0]) for s in valid_species}
    print(f"Species classes: {len(valid_species):,} (min_count={min_class_count})")
    return le, tid_to_cls, valid_species


def build_aggregation_matrices(
    taxid_to_std_ranks: dict[int, dict[str, int]],
    species_encoder: LabelEncoder,
    tid_to_cls: dict[int, int],
    ranks: list[str],
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, LabelEncoder]]:
    """
    For each rank, build:
      - A (num_groups, num_species) aggregation matrix M where
        M[group_idx, species_idx] = 1 if that species belongs to that group.
      - A LabelEncoder mapping group taxids to indices.
      - A label vector mapping species_idx -> group_idx.
    Returns dict[rank] = (agg_matrix, species_to_group_label, group_encoder).
    """
    n_species = len(species_encoder.classes_)
    result: dict[str, tuple[torch.Tensor, LabelEncoder]] = {}

    for rank in ranks:
        # Collect all group taxids for valid species at this rank
        group_taxids: set[int] = set()
        for species_tid in species_encoder.classes_:
            species_tid = int(species_tid)
            rank_dict = taxid_to_std_ranks.get(species_tid, {})
            if rank in rank_dict:
                group_taxids.add(int(rank_dict[rank]))

        if len(group_taxids) < 2:
            print(f"  Skipping rank '{rank}': only {len(group_taxids)} groups")
            continue

        group_encoder = LabelEncoder()
        group_encoder.fit(sorted(group_taxids))
        n_groups = len(group_encoder.classes_)

        # Build the aggregation matrix
        M = torch.zeros(n_groups, n_species, dtype=torch.float32, device=device)
        species_to_group = torch.full((n_species,), -1, dtype=torch.long, device=device)

        for species_tid in species_encoder.classes_:
            species_tid = int(species_tid)
            species_idx = tid_to_cls[species_tid]
            rank_dict = taxid_to_std_ranks.get(species_tid, {})
            if rank in rank_dict:
                group_tid = int(rank_dict[rank])
                group_idx = int(group_encoder.transform([group_tid])[0])
                M[group_idx, species_idx] = 1.0
                species_to_group[species_idx] = group_idx

        result[rank] = (M, species_to_group, group_encoder)
        print(f"  Rank '{rank}': {n_groups:,} groups")

    return result


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
def load_parquet_files(glob_pattern: str) -> pd.DataFrame:
    """Load and concatenate parquet files matching glob pattern."""
    import glob as _glob

    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {glob_pattern}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} OGs from {len(files)} parquet files ({glob_pattern})")
    return df


# ──────────────────────────────────────────────────────────────────────
# On-the-fly hidden extraction
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_last_protein_hidden(
    pretrained_model: nn.Module,
    seqs: list[str],
    taxids: list[int],
    tokenizer: PureARTokenizer,
    layer_idx: int,
    n_max_protein: int,
    rand_generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """
    Subsample proteins, feed through the frozen model, and return hidden
    states of the last protein at the specified layer.

    The subsampling reorders proteins randomly (matching the original
    save_last_protein_hidden.py logic), so the taxid of the last protein
    is determined *after* subsampling.

    Returns:
        hidden: (seq_len, hidden_dim) float16 tensor on cuda
        last_taxid: taxonomy id of the last protein after subsampling
    """
    n_max_protein = min(n_max_protein, len(seqs))
    idxs = torch.randperm(len(seqs), generator=rand_generator)[:n_max_protein]
    seqs = [seqs[i] for i in idxs]
    taxids = [taxids[i] for i in idxs]

    last_protein_len = len(seqs[-1])
    last_taxid = int(taxids[-1])

    input_ids = (
        tokenizer.tokenize_multi_proteins(
            proteins=seqs, flipped=False, add_sep=False, return_list=False
        )[:-1]
        .unsqueeze(0)
        .to("cuda")
    )

    output = pretrained_model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden_states = output.hidden_states
    hidden = hidden_states[layer_idx][0, -last_protein_len:, :]  # (seq_len, hidden_dim)

    del output, hidden_states
    return hidden, last_taxid


@torch.no_grad()
def extract_last_protein_hidden_batch(
    pretrained_model: nn.Module,
    seqs: list[str],
    taxids: list[int],
    tokenizer: PureARTokenizer,
    layer_idx: int,
    n_max_protein: int,
    rand_generator: torch.Generator,
    n_samples: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Draw multiple subsamples from one OG and run all through the frozen model
    in a single batched forward pass.

    Returns:
        last_hidden: (n_samples, max_last_len, hidden_dim) float16 tensor on cuda
        last_mask: (n_samples, max_last_len) bool tensor on cuda
        last_taxids: taxonomy ids for each sampled last protein
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    n_keep = min(n_max_protein, len(seqs))
    sampled_inputs: list[torch.Tensor] = []
    sampled_input_lens: list[int] = []
    sampled_last_lens: list[int] = []
    sampled_last_taxids: list[int] = []

    for _ in range(n_samples):
        idxs = torch.randperm(len(seqs), generator=rand_generator)[:n_keep]
        sampled_seqs = [seqs[i] for i in idxs]
        sampled_taxids = [taxids[i] for i in idxs]

        last_len = len(sampled_seqs[-1])
        input_ids = tokenizer.tokenize_multi_proteins(
            proteins=sampled_seqs, flipped=False, add_sep=False, return_list=False
        )[:-1]

        sampled_inputs.append(input_ids)
        sampled_input_lens.append(int(input_ids.numel()))
        sampled_last_lens.append(last_len)
        sampled_last_taxids.append(int(sampled_taxids[-1]))

    max_input_len = max(sampled_input_lens)
    batch_input_ids = torch.full(
        (n_samples, max_input_len),
        pad_token_id,
        dtype=sampled_inputs[0].dtype,
    )
    for i, input_ids in enumerate(sampled_inputs):
        batch_input_ids[i, : sampled_input_lens[i]] = input_ids

    batch_input_ids = batch_input_ids.to("cuda", non_blocking=True)
    output = pretrained_model(
        input_ids=batch_input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    hidden = output.hidden_states[layer_idx]
    max_last_len = max(sampled_last_lens)
    last_hidden = torch.zeros(
        (n_samples, max_last_len, hidden.shape[-1]),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    last_mask = torch.zeros(
        (n_samples, max_last_len),
        dtype=torch.bool,
        device=hidden.device,
    )

    for i, (input_len, last_len) in enumerate(zip(sampled_input_lens, sampled_last_lens)):
        start = input_len - last_len
        last_hidden[i, :last_len] = hidden[i, start:input_len]
        last_mask[i, :last_len] = True

    del output
    return last_hidden, last_mask, sampled_last_taxids


# ──────────────────────────────────────────────────────────────────────
# Hierarchical loss
# ──────────────────────────────────────────────────────────────────────
def hierarchical_taxonomic_loss(
    logits: torch.Tensor,
    species_label: int,
    agg_data: dict[str, tuple[torch.Tensor, torch.Tensor, LabelEncoder]],
    rank_weights: dict[str, float],
    species_weight: float,
    species_ce_weight: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute hierarchical taxonomic loss for a single protein's positions.

    Args:
        logits: (n_positions, n_species) raw logits from linear probe
        species_label: integer class index for the species
        agg_data: {rank: (agg_matrix, species_to_group_label, group_encoder)}
        rank_weights: {rank_name: loss_weight}
        species_weight: weight for the species-level CE loss
        species_ce_weight: class weights for species CE (or None)
    """
    n_positions = logits.shape[0]
    device = logits.device

    # Species-level cross-entropy
    species_labels = torch.full(
        (n_positions,), species_label, dtype=torch.long, device=device
    )
    loss = species_weight * F.cross_entropy(
        logits, species_labels, weight=species_ce_weight
    )

    # Species probabilities for aggregation up the tree
    species_probs = F.softmax(logits, dim=-1)  # (n_positions, n_species)

    for rank_name, (M, species_to_group, _) in agg_data.items():
        w = rank_weights.get(rank_name, 0.0)
        if w <= 0.0:
            continue

        group_label = species_to_group[species_label]
        if group_label < 0:
            continue

        # Aggregate species probs to group probs: (n_positions, n_groups)
        group_probs = species_probs @ M.T
        group_log_probs = torch.log(group_probs + 1e-8)

        group_labels = torch.full(
            (n_positions,), int(group_label), dtype=torch.long, device=device
        )
        loss = loss + w * F.nll_loss(group_log_probs, group_labels)

    return loss


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    df: pd.DataFrame,
    pretrained_model: nn.Module,
    probe: nn.Module,
    tokenizer: PureARTokenizer,
    tid_to_cls: dict[int, int],
    taxid_to_std_ranks: dict[int, dict[str, int]],
    agg_data: dict[str, tuple[torch.Tensor, torch.Tensor, LabelEncoder]],
    layer_idx: int,
    n_max_protein: int,
    rand_generator: torch.Generator,
    n_species: int,
) -> dict[str, float]:
    """
    Evaluate species + hierarchical accuracy on a split.
    For each OG, extract the last protein hidden and predict.
    """
    probe.eval()

    # Confusion matrix for species
    species_conf = np.zeros((n_species, n_species), dtype=np.int64)
    # Per-rank confusion matrices
    rank_confs: dict[str, np.ndarray] = {}
    for rank_name, (_, _, group_enc) in agg_data.items():
        n_groups = len(group_enc.classes_)
        rank_confs[rank_name] = np.zeros((n_groups, n_groups), dtype=np.int64)

    n_proteins_eval = 0
    n_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        seqs = row["seq"]
        taxids = row["taxid"]

        try:
            hidden, last_tid = extract_last_protein_hidden(
                pretrained_model,
                seqs,
                taxids,
                tokenizer,
                layer_idx,
                n_max_protein,
                rand_generator,
            )
            cls_idx = tid_to_cls.get(last_tid)
            if cls_idx is None:
                n_skipped += 1
                continue
            logits = probe(hidden.float())  # (seq_len, n_species)
            preds = logits.argmax(dim=1).cpu().numpy()
            species_conf[cls_idx] += np.bincount(preds, minlength=n_species)

            # Hierarchical evaluation
            species_probs = F.softmax(logits, dim=-1)
            for rank_name, (M, species_to_group, _) in agg_data.items():
                group_label = species_to_group[cls_idx].item()
                if group_label < 0:
                    continue
                group_probs = species_probs @ M.T
                group_preds = group_probs.argmax(dim=1).cpu().numpy()
                n_groups = rank_confs[rank_name].shape[0]
                rank_confs[rank_name][group_label] += np.bincount(
                    group_preds, minlength=n_groups
                )

            n_proteins_eval += 1
        except Exception as e:
            torch.cuda.empty_cache()
            n_skipped += 1
            continue

    results: dict[str, float] = {}

    # Species metrics
    total = species_conf.sum()
    results["species_overall_acc"] = (
        float(species_conf.trace() / total) if total > 0 else 0.0
    )
    class_totals = species_conf.sum(axis=1)
    valid = class_totals > 0
    per_class_acc = np.zeros(n_species, dtype=np.float64)
    per_class_acc[valid] = species_conf.diagonal()[valid] / class_totals[valid]
    results["species_balanced_acc"] = (
        float(per_class_acc[valid].mean()) if valid.any() else 0.0
    )

    # Per-rank metrics
    for rank_name, conf in rank_confs.items():
        total = conf.sum()
        results[f"{rank_name}_overall_acc"] = (
            float(conf.trace() / total) if total > 0 else 0.0
        )
        class_totals = conf.sum(axis=1)
        valid = class_totals > 0
        n_g = conf.shape[0]
        per_class_acc = np.zeros(n_g, dtype=np.float64)
        per_class_acc[valid] = conf.diagonal()[valid] / class_totals[valid]
        results[f"{rank_name}_balanced_acc"] = (
            float(per_class_acc[valid].mean()) if valid.any() else 0.0
        )

    results["n_proteins"] = float(n_proteins_eval)
    results["n_skipped"] = float(n_skipped)
    return results


# ──────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────
def save_checkpoint(
    path: Path,
    epoch: int,
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    best_val_bacc: float,
    history: list[dict],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": probe.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "best_val_bacc": best_val_bacc,
            "history": history,
            "args": vars(args),
        },
        path,
    )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def run() -> None:
    args = parse_args()
    set_seed(args.seed)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, args)

    rank_weights = parse_rank_weights(args.rank_weights)
    species_loss_weight = rank_weights.pop("species", 1.0)
    print(f"Loss weights: species={species_loss_weight}, ranks={rank_weights}")

    # ── Load data ────────────────────────────────────────────────────
    train_df = load_parquet_files(args.train_parquet_glob)
    val_df = pd.read_parquet(args.val_parquet)
    test_df = pd.read_parquet(args.test_parquet)
    print(
        f"Splits: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}"
    )

    # ── Taxonomy setup ───────────────────────────────────────────────
    taxid_to_std_ranks = load_full_taxonomy(args.taxonomy_mapping_file)
    species_encoder, tid_to_cls, valid_species = build_species_classes(
        taxid_to_std_ranks, args.min_class_count, train_df
    )
    n_species = len(species_encoder.classes_)

    print("Building aggregation matrices for hierarchical loss...")
    agg_data = build_aggregation_matrices(
        taxid_to_std_ranks, species_encoder, tid_to_cls, TAXONOMY_RANKS, device
    )

    # ── Load pretrained model ────────────────────────────────────────
    print("Loading pretrained model...")
    tokenizer = PureARTokenizer(**TOKENIZER_CONFIG)

    pretrained_model = load_convert_parent(
        tokenizer(MSA_PAD),
        RANK,
        PRETRAIN_DIR,
        load_step=-1,
        evodiff2=True,
        tokenizer=None,
        new_vocab_size=None,
        use_flash_attention_2=True,
    )
    pretrained_model = pretrained_model.half().to("cuda").eval()
    for param in pretrained_model.parameters():
        param.requires_grad = False
    print("Pretrained model loaded and frozen.")
    pad_token_id = int(tokenizer(MSA_PAD))

    # ── Build probe ──────────────────────────────────────────────────
    probe = nn.Linear(HIDDEN_DIM, n_species).to(device)

    # Class weights for species CE
    if args.class_weight_mode == "balanced" or args.class_weight_mode == "log":
        # explode taxid column to count species occurrences
        exploded = train_df.explode("taxid")
        exploded["taxid"] = exploded["taxid"].astype(int)
        exploded = exploded[exploded["taxid"].isin(valid_species)]
        exploded["cls"] = exploded["taxid"].map(tid_to_cls)
        species_counts = exploded["cls"].value_counts().to_dict()
        if args.class_weight_mode == "balanced":
            total = sum(species_counts.values())
            cw = torch.ones(n_species, dtype=torch.float32, device=device)
            for cls_idx, count in species_counts.items():
                if count > 0:
                    cw[cls_idx] = total / (n_species * count)
        else:  # log
            total = sum(species_counts.values())
            cw = torch.ones(n_species, dtype=torch.float32, device=device)
            for cls_idx, count in species_counts.items():
                if count > 0:
                    cw[cls_idx] = float(np.log1p(total / (n_species * count)))
            cw /= cw.mean()
        species_ce_weight = cw
    else:
        species_ce_weight = None

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ── Resume ───────────────────────────────────────────────────────
    checkpoint_last = output_dir / "checkpoint_last.pt"
    checkpoint_best = output_dir / "checkpoint_best.pt"
    start_epoch = 0
    best_val_bacc = -1.0
    history: list[dict] = []

    resume_path: Path | None = None
    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
    elif checkpoint_last.exists():
        resume_path = checkpoint_last

    if resume_path is not None and resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        probe.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_bacc = float(ckpt.get("best_val_bacc", -1.0))
        history = list(ckpt.get("history", []))
        print(f"Resumed at epoch {start_epoch}, best_val_bacc={best_val_bacc:.4f}")

    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    rand_generator = torch.Generator().manual_seed(args.seed)

    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        probe.train()

        # Shuffle training data
        epoch_rng = np.random.default_rng(args.seed + epoch)
        shuffled_indices = epoch_rng.permutation(len(train_df))

        epoch_loss = 0.0
        epoch_positions = 0
        n_ogs = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(shuffled_indices, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step_i, og_idx in enumerate(pbar):
            row = train_df.iloc[int(og_idx)]
            seqs = row["seq"]
            taxids = row["taxid"]

            try:
                hidden_batch, hidden_mask, last_tids = extract_last_protein_hidden_batch(
                    pretrained_model,
                    seqs,
                    taxids,
                    tokenizer,
                    args.layer_idx,
                    args.n_max_protein,
                    rand_generator,
                    max(1, args.n_sample_per_og_train),
                    pad_token_id,
                )
                cls_indices = [tid_to_cls.get(tid) for tid in last_tids]
                valid_sample_ids = [i for i, cls_idx in enumerate(cls_indices) if cls_idx is not None]
                if not valid_sample_ids:
                    continue
                hidden_batch = hidden_batch[valid_sample_ids]
                hidden_mask = hidden_mask[valid_sample_ids]
                valid_cls_indices = [int(cls_indices[i]) for i in valid_sample_ids]

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    flat_hidden = hidden_batch[hidden_mask]  # (sum_last_lens, hidden_dim)
                    flat_logits = probe(flat_hidden.float())

                    sample_lens = hidden_mask.sum(dim=1).tolist()
                    offset = 0
                    total_loss = flat_logits.new_zeros(())
                    total_weighted_loss = 0.0
                    n_pos = 0
                    for sample_len, cls_idx in zip(sample_lens, valid_cls_indices):
                        logits = flat_logits[offset : offset + sample_len]
                        offset += sample_len
                        sample_loss = hierarchical_taxonomic_loss(
                            logits=logits,
                            species_label=cls_idx,
                            agg_data=agg_data,
                            rank_weights=rank_weights,
                            species_weight=species_loss_weight,
                            species_ce_weight=species_ce_weight,
                        )
                        total_loss = total_loss + sample_loss
                        total_weighted_loss += float(sample_loss.detach()) * sample_len
                        n_pos += sample_len
                    loss = total_loss / args.accumulation_steps

                if amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += total_weighted_loss
                epoch_positions += n_pos
                n_ogs += 1

                # Optimizer step every accumulation_steps
                if (step_i + 1) % args.accumulation_steps == 0:
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if n_ogs % 100 == 0:
                    avg_loss = epoch_loss / max(epoch_positions, 1)
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}", pos=f"{epoch_positions:,}", ogs=n_ogs
                    )

            except Exception as e:
                torch.cuda.empty_cache()
                if n_ogs < 5:
                    print(f"Error on OG {row.get('og', '?')}: {e}")
                continue

        # Flush remaining gradients
        if n_ogs % args.accumulation_steps != 0:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = epoch_loss / max(epoch_positions, 1)
        if scheduler is not None:
            scheduler.step(train_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - t0

        # ── Evaluation ───────────────────────────────────────────────
        should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
        val_metrics: dict[str, float] | None = None
        improved = False

        if should_eval:
            val_metrics = evaluate(
                df=val_df,
                pretrained_model=pretrained_model,
                probe=probe,
                tokenizer=tokenizer,
                tid_to_cls=tid_to_cls,
                taxid_to_std_ranks=taxid_to_std_ranks,
                agg_data=agg_data,
                layer_idx=args.layer_idx,
                n_max_protein=args.n_max_protein,
                rand_generator=torch.Generator().manual_seed(args.seed),
                n_species=n_species,
            )
            val_bacc = val_metrics["species_balanced_acc"]
            improved = val_bacc > best_val_bacc
            if improved:
                best_val_bacc = val_bacc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in probe.state_dict().items()
                }

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_positions": epoch_positions,
            "train_ogs": n_ogs,
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        }
        if val_metrics is not None:
            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            epoch_metrics["best_val_species_balanced_acc"] = best_val_bacc
        history.append(epoch_metrics)

        # Print summary
        summary = (
            f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | "
            f"lr={current_lr:.3e} | ogs={n_ogs:,} | pos={epoch_positions:,} | "
            f"time={epoch_time:.1f}s"
        )
        if val_metrics is not None:
            summary += (
                f" | sp_bacc={val_metrics['species_balanced_acc']:.4f}"
                f" | sp_acc={val_metrics['species_overall_acc']:.4f}"
            )
            for rank_name in TAXONOMY_RANKS:
                key = f"{rank_name}_balanced_acc"
                if key in val_metrics:
                    summary += f" | {rank_name[:3]}_bacc={val_metrics[key]:.4f}"
        print(summary)

        # Checkpointing
        save_checkpoint(
            checkpoint_last,
            epoch,
            probe,
            optimizer,
            scheduler,
            best_val_bacc,
            history,
            args,
        )
        if improved:
            save_checkpoint(
                checkpoint_best,
                epoch,
                probe,
                optimizer,
                scheduler,
                best_val_bacc,
                history,
                args,
            )
        if args.save_every_epoch:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch}.pt",
                epoch,
                probe,
                optimizer,
                scheduler,
                best_val_bacc,
                history,
                args,
            )

    # ── Final evaluation with best weights ───────────────────────────
    probe.load_state_dict(best_state)

    print("\nFinal evaluation with best model...")
    final_rand_gen = torch.Generator().manual_seed(args.seed)
    val_metrics = evaluate(
        df=val_df,
        pretrained_model=pretrained_model,
        probe=probe,
        tokenizer=tokenizer,
        tid_to_cls=tid_to_cls,
        taxid_to_std_ranks=taxid_to_std_ranks,
        agg_data=agg_data,
        layer_idx=args.layer_idx,
        n_max_protein=args.n_max_protein,
        rand_generator=final_rand_gen,
        n_species=n_species,
    )
    test_metrics = evaluate(
        df=test_df,
        pretrained_model=pretrained_model,
        probe=probe,
        tokenizer=tokenizer,
        tid_to_cls=tid_to_cls,
        taxid_to_std_ranks=taxid_to_std_ranks,
        agg_data=agg_data,
        layer_idx=args.layer_idx,
        n_max_protein=args.n_max_protein,
        rand_generator=torch.Generator().manual_seed(args.seed),
        n_species=n_species,
    )

    print("\n=== Final Results (best checkpoint) ===")
    for split_name, metrics in [("val", val_metrics), ("test", test_metrics)]:
        print(f"\n{split_name.upper()}:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Save artifacts ───────────────────────────────────────────────
    model_path = output_dir / "probe_hierarchical_model.pt"
    torch.save(probe.state_dict(), model_path)

    final_metrics = {
        "n_species": n_species,
        "hidden_dim": HIDDEN_DIM,
        "rank_weights": {**rank_weights, "species": species_loss_weight},
        "val": val_metrics,
        "test": test_metrics,
        "best_val_species_balanced_acc": best_val_bacc,
        "history": history,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save class encoders
    encoder_data = {
        "species_encoder": species_encoder,
        "tid_to_cls": tid_to_cls,
        "agg_data_encoders": {
            rank: group_enc for rank, (_, _, group_enc) in agg_data.items()
        },
    }
    with open(output_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoder_data, f)

    print(f"\nArtifacts saved to {output_dir}")


if __name__ == "__main__":
    run()

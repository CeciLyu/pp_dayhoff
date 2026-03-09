"""
Seq position and protein family evalutaion of probe accuracy

For each protein family, randomly sample proteins from single class
and multi class. Run model and get hidden at layer of interest,
run probe on last protein.

Save per position accuracy.
"""

import itertools
import os
import pickle
import torch
import torch.nn.functional as F
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from deimm.model.tokenizer import PureARTokenizer
from deimm.utils.training_utils import (
    load_convert_parent,
    seed_everything,
)
from deimm.utils.constants import (
    MSA_PAD,
    PROTEIN_SEP,
    RANK,
    PRETRAIN_DIR,
)

# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════
TEST_FPATH = "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet"


# RANKS_TO_PROBE = ["domain", "phylum", "class", "order", "family", "genus"]
RANKS_TO_PROBE = [
    "phylum",
    "domain",
    "class",
    "order",
    "family",
    "genus",
]  # start with one rank for speed; expand after initial test
PROBE_PATH = Path("/scratch/suyuelyu/deimm/results/probe_taxon/")

# Layer → probe file mapping per rank
# Format: {rank: {layer_idx: path_to_probe}}
# Adjust paths to match your setup
LAYER_PROBE_MAP = {
    "phylum": {
        16: PROBE_PATH / "phylum_ce_mmap_lyr16" / "probe_phylum_data.pkl",
    },
    "domain": {
        16: PROBE_PATH / "domain_ce_mmap_lyr16" / "probe_domain_data.pkl",
    },
    "class": {
        16: PROBE_PATH / "class_ce_mmap_lyr16" / "probe_class_data.pkl",
    },
    "order": {
        16: PROBE_PATH / "order_ce_mmap_lyr16" / "probe_order_data.pkl",
    },
    "family": {
        16: PROBE_PATH / "family_ce_mmap_lyr16" / "probe_family_data.pkl",
    },
    "genus": {
        16: PROBE_PATH / "genus_ce_mmap_lyr16" / "probe_genus_data.pkl",
    },
}

STEER_LAYERS = [16]  # start with one layer for speed; expand after initial test

SEED = 3525
N_SAMPLES_PER_OG = 10  # number of generated sequences per condition
MAX_CONTEXT_PROTEINS = 64  # max proteins in context (last slot = generation)
TAXONOMY_MAPPING_FILE = "/scratch/suyuelyu/deimm/data/oma/taxid_to_std_ranks.pkl"

OUTPUT_DIR = PROBE_PATH / "per_pos_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda"
SAVE_EVERY_OG = 5

seed_everything(SEED)


def load_dayhoff_model_tokenizer() -> tuple[torch.nn.Module, PureARTokenizer]:
    tokenizer_config = {
        "vocab_path": "vocab_UL_ALPHABET_PLUS.txt",
        "allow_unk": False,
    }
    tokenizer = PureARTokenizer(**tokenizer_config)
    model = load_convert_parent(
        tokenizer(MSA_PAD),
        RANK,
        PRETRAIN_DIR,
        load_step=-1,
        evodiff2=True,
        tokenizer=None,
        new_vocab_size=None,
        use_flash_attention_2=True,
    )
    model = model.half().to(DEVICE)
    model = model.eval()
    print("Model loaded")
    return model, tokenizer


def load_linear_probe(probe_path: str | Path) -> dict:
    print(f"  Loading probe from {probe_path}...")
    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)
    linear_weights = torch.from_numpy(probe_data["W"]).float().to(DEVICE)
    return {
        "linear_weights_t": linear_weights.T.contiguous(),
        "intercept": torch.from_numpy(probe_data["intercept"]).float().to(DEVICE),
        "species_to_rank": probe_data["rank_mapping"],
        "rank_to_classidx": {lab: idx for idx, lab in enumerate(probe_data["classes"])},
        "classidx_to_rank": {idx: lab for idx, lab in enumerate(probe_data["classes"])},
        "classes": list(probe_data["classes"]),
        "n_classes": len(probe_data["classes"]),
    }


def load_taxonomy_mapping(
    mapping_file: str, ranks: list[str]
) -> dict[str, dict[int, int]]:
    with open(mapping_file, "rb") as f:
        taxid_to_std_ranks = pickle.load(f)
    rank_mapping = {r: {} for r in ranks}
    for species_tid, rank_dict in taxid_to_std_ranks.items():
        for rank in ranks:
            if rank in rank_dict:
                rank_mapping[rank][int(species_tid)] = int(rank_dict[rank])
    return rank_mapping


def sample_proteins_for_og(
    rank_ids, src_rank_id=None, tgt_rank_id=None, rank_to_indices=None
):
    # if src_rank_id or tgt_rank_id is provided, filter to those rank ids first
    if rank_to_indices is None:
        rank_to_indices = defaultdict(list)
        for i, r in enumerate(rank_ids):
            rank_to_indices[r].append(i)
    rank_idxs = list(range(len(rank_ids)))
    if src_rank_id is not None and tgt_rank_id is not None:
        if src_rank_id != tgt_rank_id:
            src_idxs = rank_to_indices.get(src_rank_id, [])
            if len(src_idxs) == 0:
                raise ValueError(f"No proteins found with rank id {src_rank_id}")
            tgt_idxs = rank_to_indices.get(tgt_rank_id, [])
            if len(tgt_idxs) == 0:
                raise ValueError(f"No proteins found with rank id {tgt_rank_id}")

            # if src == tgt, sample from that rank; else, sample src and tgt separately
            # evaluating from one class to another
            n_src_proteins = random.randint(
                1, min(MAX_CONTEXT_PROTEINS - 1, len(src_idxs))
            )
            # src_idxs_sampled = no replacement sampling from src_idxs
            src_idxs_sampled = random.sample(
                src_idxs, min(n_src_proteins, len(src_idxs))
            )
            # tgt_idxs_sampled = 1 random sample from tgt_idxs
            tgt_idxs_sampled = random.sample(tgt_idxs, 1)
            idxs_sampled = src_idxs_sampled + tgt_idxs_sampled
            return idxs_sampled
        else:
            # evaluating within one class, sample all from that class
            rank_idxs = rank_to_indices.get(src_rank_id, [])
            if len(rank_idxs) <= 1:
                print(
                    f"  WARNING: Only {len(rank_idxs)} proteins found with rank id {src_rank_id}"
                )
                return None

    max_n_proteins = min(MAX_CONTEXT_PROTEINS, len(rank_idxs))
    if max_n_proteins < 2:
        return None
    n_proteins = random.randint(2, max_n_proteins)
    idxs_sampled = random.sample(rank_idxs, n_proteins)
    return idxs_sampled


def sample_context_for_og(
    protein_names: list,
    rank_ids: list,
    seqs: list,
    tokenizer: PureARTokenizer,
    src_rank_id: int | None = None,
    tgt_rank_id: int | None = None,
    rank_to_indices: dict | None = None,
) -> dict | None:
    idxs = sample_proteins_for_og(
        rank_ids=rank_ids,
        src_rank_id=src_rank_id,
        tgt_rank_id=tgt_rank_id,
        rank_to_indices=rank_to_indices,
    )
    if idxs is None:
        return None

    seqs_sampled = [seqs[i] for i in idxs]
    input_ids = tokenizer.tokenize_multi_proteins(
        proteins=seqs_sampled, flipped=False, add_sep=False, return_list=False
    )[:-1]
    return {
        "protein_names": [protein_names[i] for i in idxs],
        "rank_ids": [rank_ids[i] for i in idxs],
        "input_ids": input_ids,
        "input_len": int(input_ids.shape[0]),
        "last_protein_len": len(seqs_sampled[-1]),
    }


def build_padded_batch(
    sample_contexts: list[dict], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(sample_contexts) == 0:
        raise ValueError("Cannot build padded batch from empty contexts")

    batch_size = len(sample_contexts)
    max_len = max(s["input_len"] for s in sample_contexts)
    input_ids = torch.full(
        (batch_size, max_len), pad_token_id, dtype=torch.long, device=DEVICE
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=DEVICE)

    for i, sample in enumerate(sample_contexts):
        seq_len = sample["input_len"]
        sample_ids = sample["input_ids"].to(DEVICE)
        input_ids[i, :seq_len] = sample_ids
        attention_mask[i, :seq_len] = 1

    return input_ids, attention_mask


def hidden_idx_to_hook_layer(layer_idx: int, n_model_layers: int) -> int:
    """
    hidden_states[k] (k>0) corresponds to output of model.model.layers[k-1].
    """
    if layer_idx <= 0 or layer_idx > n_model_layers:
        raise ValueError(
            f"Unsupported hidden state index {layer_idx}; expected 1..{n_model_layers}"
        )
    return layer_idx - 1


@torch.inference_mode()
def probe_on_last_protein_hidden(
    probe: dict,
    last_protein_hidden: torch.Tensor,
) -> list:
    # argmax(softmax(logits)) == argmax(logits), so skip softmax for speed.
    logits = F.linear(
        last_protein_hidden.float(),
        probe["linear_weights_t"],
        probe["intercept"],
    )
    pred_class_idx = torch.argmax(logits, dim=-1)  # shape: (seq_len,)
    classes = probe["classes"]
    pred_tgt_rank_id = [classes[idx] for idx in pred_class_idx.tolist()]
    return pred_tgt_rank_id


@torch.inference_mode()
def get_last_protein_hidden_for_samples_batched(
    model: torch.nn.Module,
    protein_names: list,
    rank_ids: list,
    seqs: list,
    tokenizer: PureARTokenizer,
    n_samples: int,
    pad_token_id: int,
    src_rank_id: int | None = None,
    tgt_rank_id: int | None = None,
    rank_to_indices: dict | None = None,
    layers: list | None = None,
) -> list[tuple]:
    if layers is None:
        raise ValueError("`layers` must be provided")

    sample_contexts = []
    for _ in range(n_samples):
        sample = sample_context_for_og(
            protein_names=protein_names,
            rank_ids=rank_ids,
            seqs=seqs,
            tokenizer=tokenizer,
            src_rank_id=src_rank_id,
            tgt_rank_id=tgt_rank_id,
            rank_to_indices=rank_to_indices,
        )
        if sample is not None:
            sample_contexts.append(sample)

    if not sample_contexts:
        return []

    input_ids, attention_mask = build_padded_batch(sample_contexts, pad_token_id)

    # Use Jamba backbone directly to avoid LM head logits computation.
    # Capture only requested hidden states via hooks instead of materializing all layers.
    backbone = model.model if hasattr(model, "model") else model
    last_hidden_by_layer = {}
    if hasattr(backbone, "layers"):
        hooks = []

        def _make_capture_hook(layer_idx):
            def _hook(_module, _input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                last_hidden_by_layer[layer_idx] = hidden

            return _hook

        try:
            n_model_layers = len(backbone.layers)
            for layer_idx in layers:
                if layer_idx == -1:
                    continue
                hook_layer = hidden_idx_to_hook_layer(layer_idx, n_model_layers)
                hooks.append(
                    backbone.layers[hook_layer].register_forward_hook(
                        _make_capture_hook(layer_idx)
                    )
                )

            output = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            if -1 in layers:
                last_hidden_by_layer[-1] = output.last_hidden_state
        finally:
            for h in hooks:
                h.remove()
    else:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states
        for layer_idx in layers:
            last_hidden_by_layer[layer_idx] = output[layer_idx]

    outputs = []
    for bidx, sample in enumerate(sample_contexts):
        seq_len = sample["input_len"]
        last_protein_len = sample["last_protein_len"]
        last_protein_hiddens = []
        for layer_idx in layers:
            hidden = last_hidden_by_layer[layer_idx]
            last_hidden = hidden[bidx, seq_len - last_protein_len : seq_len, :]
            last_protein_hiddens.append(last_hidden)
        outputs.append(
            (
                sample["protein_names"],
                last_protein_hiddens,
                sample["rank_ids"],
            )
        )
    return outputs


def main():
    # Load model and tokenizer
    model, tokenizer = load_dayhoff_model_tokenizer()
    pad_token_id = tokenizer(MSA_PAD)

    # Load taxonomy mapping
    taxid_to_rank = load_taxonomy_mapping(TAXONOMY_MAPPING_FILE, RANKS_TO_PROBE)

    # Load probes
    all_probes = {}  # {rank: {layer: probe_dict}}
    for rank in RANKS_TO_PROBE:
        if rank not in LAYER_PROBE_MAP:
            print(f"No probes configured for rank {rank}, skipping")
            continue
        all_probes[rank] = {}
        for layer_idx, path in LAYER_PROBE_MAP[rank].items():
            if layer_idx not in STEER_LAYERS:
                continue
            if path.exists():
                all_probes[rank][layer_idx] = load_linear_probe(path)
            else:
                print(f"  WARNING: {path} not found")

    # Load test data
    meta_grouped_test = pd.read_parquet(TEST_FPATH)
    print(f"Loaded test data with {len(meta_grouped_test)} orthologous groups")

    # loop through meta_grouped_test
    for rank in RANKS_TO_PROBE:
        if rank not in all_probes or not all_probes[rank]:
            print(f"No loaded probes for rank {rank}, skipping...")
            continue
        layers_for_rank = [lyr for lyr in STEER_LAYERS if lyr in all_probes[rank]]
        if not layers_for_rank:
            print(f"No matching layers configured for rank {rank}, skipping...")
            continue

        output_path = OUTPUT_DIR / f"probe_taxon_per_pos_test_results_{rank}.pkl"
        rank_results = []
        # check if output_path already exists, if so, resume from there
        if output_path.exists():
            print(f"Resuming from existing results at {output_path}...")
            with open(output_path, "rb") as f:
                loaded_results = pickle.load(f)
            # Backward compatible with old files that may contain mixed ranks.
            rank_results = [r for r in loaded_results if r.get("rank") == rank]

        # if contains all OGs, skip evaluation for this rank
        ogs_evaluated = {r["og"] for r in rank_results}
        if len(ogs_evaluated) == len(meta_grouped_test):
            print(f"All OGs already evaluated for rank {rank}, skipping...")
            continue

        print(f"Evaluating rank {rank}...")
        print(f"saving results to {output_path}...")
        n_new_og_results = 0
        for row in tqdm(
            meta_grouped_test.itertuples(index=False), total=len(meta_grouped_test)
        ):
            if row.og in ogs_evaluated:
                continue
            # n_classes for current rank in this OG
            taxids = row.taxid
            rank_ids = [taxid_to_rank[rank].get(tid, None) for tid in taxids]
            rank_to_indices = defaultdict(list)
            for idx, rank_id in enumerate(rank_ids):
                rank_to_indices[rank_id].append(idx)
            n_classes = len(set(c for c in rank_ids if c is not None))
            if n_classes < 2:
                sample_outputs = get_last_protein_hidden_for_samples_batched(
                    model,
                    row.protein,
                    rank_ids,
                    row.seq,
                    tokenizer,
                    n_samples=N_SAMPLES_PER_OG,
                    pad_token_id=pad_token_id,
                    src_rank_id=None,
                    tgt_rank_id=None,
                    rank_to_indices=rank_to_indices,
                    layers=layers_for_rank,
                )
                for (
                    protein_names,
                    last_protein_hiddens,
                    rank_ids_sampled,
                ) in sample_outputs:
                    # run probe on last protein hidden
                    for layer_idx, last_protein_hidden in zip(
                        layers_for_rank, last_protein_hiddens
                    ):
                        probe = all_probes[rank][layer_idx]
                        pred_tgt_rank_id = probe_on_last_protein_hidden(
                            probe, last_protein_hidden
                        )
                        rank_results.append(
                            {
                                "og": row.og,
                                "rank": rank,
                                "protein_names": protein_names,
                                "rank_ids": rank_ids_sampled,
                                "layer": layer_idx,
                                "src_rank_id": None,
                                "tgt_rank_id": None,
                                "pred_tgt_rank_id": pred_tgt_rank_id,
                            }
                        )
            else:
                # for each pair of classes in this OG, sample proteins and run probe
                unique_rank_ids = set(c for c in rank_ids if c is not None)
                # show inner loop progress with another tqdm bar
                for src_rank_id, tgt_rank_id in tqdm(
                    itertools.product(unique_rank_ids, repeat=2),
                    total=n_classes**2,
                    desc="Sampling class pairs for evaluation",
                ):
                    sample_outputs = get_last_protein_hidden_for_samples_batched(
                        model,
                        row.protein,
                        rank_ids,
                        row.seq,
                        tokenizer,
                        n_samples=N_SAMPLES_PER_OG,
                        pad_token_id=pad_token_id,
                        src_rank_id=src_rank_id,
                        tgt_rank_id=tgt_rank_id,
                        rank_to_indices=rank_to_indices,
                        layers=layers_for_rank,
                    )
                    for (
                        protein_names,
                        last_protein_hiddens,
                        rank_ids_sampled,
                    ) in sample_outputs:
                        # run probe on last protein hidden
                        for layer_idx, last_protein_hidden in zip(
                            layers_for_rank, last_protein_hiddens
                        ):
                            probe = all_probes[rank][layer_idx]
                            pred_tgt_rank_id = probe_on_last_protein_hidden(
                                probe, last_protein_hidden
                            )
                            rank_results.append(
                                {
                                    "og": row.og,
                                    "rank": rank,
                                    "protein_names": protein_names,
                                    "rank_ids": rank_ids_sampled,
                                    "layer": layer_idx,
                                    "src_rank_id": src_rank_id,
                                    "tgt_rank_id": tgt_rank_id,
                                    "pred_tgt_rank_id": pred_tgt_rank_id,
                                }
                            )
            n_new_og_results += 1
            if n_new_og_results % SAVE_EVERY_OG == 0:
                with open(output_path, "wb") as f:
                    pickle.dump(rank_results, f)

        # Save all results to a pickle file
        with open(output_path, "wb") as f:
            pickle.dump(rank_results, f)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

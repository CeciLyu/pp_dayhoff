
import re
import pickle
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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
RANKS_TO_PROBE = ["phylum"]  # start with one rank for speed; expand after initial test
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
}

STEER_LAYERS = [16]  # start with one layer for speed; expand after initial test
STEER_ALPHA = [
    25.0,
    50.0,
    # 0.5,
    # 1.0,
    # 1.5,
]  # fewer alphas for generation (slower than PPL)

SEED = 3525
N_SAMPLES_PER_CONDITION = 3  # number of generated sequences per condition
N_TEST_OGS = 50  # max OGs to test (None = all)
MAX_CONTEXT_PROTEINS = 63  # max proteins in context (last slot = generation)
MAX_GEN_LEN = 1024  # max generation length
TEMPERATURE = 1.0  # sampling temperature
TOP_K = 0  # 0 = no top-k filtering
TOP_P = 0.95  # nucleus sampling threshold
MIN_SEQS_PER_CLASS = 2  # need ≥2 seqs in source class (1 context + 1 baseline)
DO_GEN_NEUTRAL = True  # whether to run neutral (no-steering) generation
DO_GEN_WRONG = False  # whether to run wrong-control (steer-to-source) generation

TAXONOMY_MAPPING_FILE = "/scratch/suyuelyu/deimm/data/oma/taxid_to_std_ranks.pkl"

OUTPUT_DIR = PROBE_PATH / "steer_generation"

DEVICE = "cuda"

seed_everything(SEED)


# ═══════════════════════════════════════════════
# BLOSUM SCORING
# ═══════════════════════════════════════════════


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    """Remove any characters not in the standard amino acid alphabet."""
    if PROTEIN_SEP in seq:
        seq = seq.split(PROTEIN_SEP)[0]  # take last protein if multiple
    return "".join(c for c in seq.upper() if c in VALID_AA)


BLOSUM62 = substitution_matrices.load("BLOSUM62")

aligner = PairwiseAligner()
aligner.substitution_matrix = BLOSUM62
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.1
aligner.mode = "local"


def blosum_similarity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    score = aligner.score(seq1, seq2)
    # aln_len = max(len(seq1), len(seq2))
    # return score / aln_len if aln_len > 0 else 0.0
    return score / len(seq2)


def sequence_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    alignments = aligner.align(seq1, seq2)
    aln = alignments[0]
    aligned = aln.format().split("\n")
    # Count matches from alignment
    seq_a = aligned[0]
    seq_b = aligned[2]
    matches = sum(a == b and a != "-" for a, b in zip(seq_a, seq_b))
    aln_len = sum(1 for a, b in zip(seq_a, seq_b) if a != "-" or b != "-")
    return matches / aln_len if aln_len > 0 else 0.0


def write_fasta_record(
    fasta_handle,
    header: str,
    seq: str,
) -> None:
    # Biopython-based FASTA serialization
    record = SeqRecord(Seq(seq), id=header, description="")
    SeqIO.write(record, fasta_handle, "fasta")


# ═══════════════════════════════════════════════
# MODEL / PROBE LOADING
# ═══════════════════════════════════════════════


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
    return {
        "linear_weights": torch.from_numpy(probe_data["W"]).float().to(DEVICE),
        "intercept": torch.from_numpy(probe_data["intercept"]).float().to(DEVICE),
        "species_to_rank": probe_data["rank_mapping"],
        "rank_to_classidx": {lab: idx for idx, lab in enumerate(probe_data["classes"])},
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


# ═══════════════════════════════════════════════
# STEERING
# ═══════════════════════════════════════════════


def hidden_idx_to_hook_layer(layer_idx: int) -> int:
    """hidden_states[k] = output of model.model.layers[k-1]."""
    return layer_idx - 1


class SteeringHook:
    """Forward hook that adds steering vector to specified positions."""

    def __init__(self, steering_vector: torch.Tensor, n_positions: int):
        """
        Args:
            steering_vector: [hidden_dim] or [n_positions, hidden_dim]
            n_positions: how many trailing positions to steer
        """
        self.steering_vector = steering_vector.half().to(DEVICE)
        self.n_positions = n_positions

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        n = self.n_positions
        # Steer positions that correspond to generated tokens so far
        # During generation, sequence grows; steer only the last n positions
        if hidden.shape[1] >= n + 1:
            hidden[:, -(n + 1) : -1, :] += self.steering_vector
        else:
            # Sequence shorter than expected, steer everything except first token
            hidden[:, 1:, :] += self.steering_vector[-hidden.shape[1] + 1 :]

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


class GenerationSteeringHook:
    """
    Simpler hook for autoregressive generation.
    Adds a fixed steering vector to the LAST position at each generation step.
    """

    def __init__(self, steering_vector_per_pos: torch.Tensor):
        """
        Args:
            steering_vector_per_pos: [hidden_dim] fixed vector added at each step
        """
        self.steer = steering_vector_per_pos.half().to(DEVICE)

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Always steer the last position (current generation step)
        hidden[:, -1, :] += self.steer

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


# class GenerationSteeringHook:
#     def __init__(self, steering_vector_per_pos):
#         self.steer = steering_vector_per_pos.half().to(DEVICE)
#         self.call_count = 0

#     def __call__(self, module, input, output):
#         self.call_count += 1

#         # Debug: print output structure on first call
#         if self.call_count == 1:
#             print(f"  Hook fired. Output type: {type(output)}")
#             if isinstance(output, tuple):
#                 print(f"  Tuple length: {len(output)}")
#                 for i, o in enumerate(output):
#                     if isinstance(o, torch.Tensor):
#                         print(f"    [{i}] tensor shape={o.shape} dtype={o.dtype}")
#                     else:
#                         print(f"    [{i}] {type(o)}")
#             elif isinstance(output, torch.Tensor):
#                 print(f"  Tensor shape={output.shape}")

#         if isinstance(output, tuple):
#             hidden = output[0].clone()
#         else:
#             hidden = output

#         before = hidden[:, -1, :].clone()
#         hidden[:, -1, :] += self.steer
#         diff = (hidden[:, -1, :] - before).abs().max().item()

#         if self.call_count == 1:
#             print(f"  Steer norm: {self.steer.norm():.4f}")
#             print(f"  Hidden norm: {before.norm():.4f}")
#             print(f"  Actual diff after modify: {diff:.6f}")

#         if isinstance(output, tuple):
#             return (hidden,) + output[1:]
#         return hidden


# ═══════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════


def compute_mean_steering_direction(
    probe: dict,
    target_class_idx: int,
    alpha: float,
) -> torch.Tensor:
    """
    Compute a fixed steering direction for generation.
    Uses the probe gradient at the class centroid (approximated as the
    probe column itself), giving a position-independent direction.

    Returns: [hidden_dim] steering vector
    """
    W = probe["linear_weights"]  # [hidden_dim, n_classes]
    b = probe["intercept"]  # [n_classes]

    # Use the target class column as an approximate "typical" hidden state direction
    # The gradient at any point h is: (softmax(hW+b) - e_target) @ W^T
    # For a fixed direction independent of h, use -W[:, target] (activation addition)
    # This is simpler and more stable for generation than adaptive gradient
    direction = W[:, target_class_idx]  # [hidden_dim]
    direction = direction / direction.norm()  # normalize
    return alpha * direction


def compute_adaptive_steering_vector(
    hidden: torch.Tensor,
    probe: dict,
    target_class_idx: int,
    alpha: float,
) -> torch.Tensor:
    """
    Compute gradient-based adaptive steering vector.

    Args:
        hidden: [seq_len, hidden_dim] current hidden states
        probe: probe dict with linear_weights and intercept
        target_class_idx: index of target class
        alpha: steering strength

    Returns: [seq_len, hidden_dim] steering vectors
    """
    W = probe["linear_weights"]
    b = probe["intercept"]
    h = hidden.float().to(DEVICE)
    prob = torch.softmax(h @ W + b, dim=-1)
    e_target = torch.zeros(prob.shape[-1], device=DEVICE)
    e_target[target_class_idx] = 1.0
    grad = (prob - e_target) @ W.t()
    return -alpha * grad


@torch.inference_mode()
def generate_with_steering(
    model: torch.nn.Module,
    tokenizer: PureARTokenizer,
    context_input_ids: torch.Tensor,
    layer_idx: int,
    steering_vector: torch.Tensor | None,
    max_len: int = MAX_GEN_LEN,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    eos_token_id: int | None = None,
) -> str:
    """
    Autoregressive generation with optional steering hook.

    Args:
        context_input_ids: [1, context_len] tokenized context
        layer_idx: which layer to hook (hidden_states index)
        steering_vector: [hidden_dim] fixed direction, or None for no steering
        max_len: maximum tokens to generate
    Returns:
        generated protein sequence as string
    """
    input_ids = context_input_ids.clone()
    generated_tokens = []

    # Set up persistent hook if steering
    handle = None
    if steering_vector is not None:
        hook_layer = hidden_idx_to_hook_layer(layer_idx)
        hook = GenerationSteeringHook(steering_vector)
        handle = model.model.layers[hook_layer].register_forward_hook(hook)

    try:
        for _ in range(max_len):
            output = model(input_ids=input_ids, return_dict=True)
            next_logits = output.logits[0, -1, :].float()

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[-1]] = -float("inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = -float("inf")

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Check EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    finally:
        if handle is not None:
            handle.remove()

    # Decode tokens to sequence
    generated_seq = tokenizer.detokenize_protein(generated_tokens)
    generated_seq = clean_sequence(generated_seq)
    return generated_seq


# ═══════════════════════════════════════════════
# OG PREPARATION
# ═══════════════════════════════════════════════


def prepare_og_for_rank(
    row: pd.Series,
    rank_mapping: dict[int, int],
    valid_classes: set | None = None,
) -> dict[int, list[tuple[str, str, int]]]:
    """
    Group proteins in an OG by their class at the given rank.

    Returns: {class_label: [(protein_name, sequence, species_taxid), ...]}
    """
    groups = defaultdict(list)
    proteins = row["protein"]
    seqs = row["seq"]
    taxids = row["taxid"]

    for name, seq, tid in zip(proteins, seqs, taxids):
        tid = int(tid)
        if tid not in rank_mapping:
            continue
        cls = rank_mapping[tid]
        if valid_classes is not None and cls not in valid_classes:
            continue
        groups[cls].append((name, seq, tid))

    return dict(groups)


def select_source_target(
    groups: dict[int, list],
    min_seqs: int = MIN_SEQS_PER_CLASS,
    rng: random.Random = None,
) -> tuple[int, int, list, tuple] | None:
    """
    Select source and target classes from grouped OG.

    Source class needs ≥ min_seqs sequences (for context + source baseline).
    Target class needs ≥ 1 sequence.

    Returns: (source_class, target_class, source_proteins, target_protein)
             or None if not possible
    """
    if rng is None:
        rng = random.Random()

    # Find classes with enough sequences for source
    source_candidates = [c for c, prots in groups.items() if len(prots) >= min_seqs]
    if not source_candidates:
        return None

    # Need at least 2 classes
    all_classes = list(groups.keys())
    if len(all_classes) < 2:
        return None

    source_class = rng.choice(source_candidates)
    target_candidates = [c for c in all_classes if c != source_class]
    if not target_candidates:
        return None
    target_class = rng.choice(target_candidates)

    source_proteins = groups[source_class]
    target_protein = rng.choice(groups[target_class])

    return source_class, target_class, source_proteins, target_protein


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_dayhoff_model_tokenizer()

    # Try to find EOS token
    eos_token_id = tokenizer.tokenizer_dict[PROTEIN_SEP]
    # Adjust if your tokenizer has a specific EOS token
    # eos_token_id = tokenizer.eos_token_id

    # Load taxonomy mapping
    tax_mapping = load_taxonomy_mapping(TAXONOMY_MAPPING_FILE, RANKS_TO_PROBE)

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
    print(f"Loaded {len(meta_grouped_test)} test OGs")

    rng = random.Random(SEED)
    all_results = []
    fasta_path = OUTPUT_DIR / f"generation_sequences_seed{SEED}.fasta"
    fasta_handle = open(fasta_path, "w")

    for rank in RANKS_TO_PROBE:
        if rank not in all_probes or not all_probes[rank]:
            continue

        print(f"\n{'═' * 60}")
        print(f"RANK: {rank}")
        print(f"{'═' * 60}")

        rank_map = tax_mapping[rank]

        # Get valid classes from probe
        first_layer = next(iter(all_probes[rank]))
        probe_classes = set(all_probes[rank][first_layer]["rank_to_classidx"].keys())

        # Find eligible OGs
        eligible_ogs = []
        for _, row in meta_grouped_test.iterrows():
            groups = prepare_og_for_rank(row, rank_map, valid_classes=probe_classes)
            result = select_source_target(groups, MIN_SEQS_PER_CLASS, rng=rng)
            if result is not None:
                eligible_ogs.append((row, groups, result))

        print(f"  {len(eligible_ogs)} eligible OGs (have ≥2 classes with enough seqs)")

        if N_TEST_OGS is not None and len(eligible_ogs) > N_TEST_OGS:
            rng.shuffle(eligible_ogs)
            eligible_ogs = eligible_ogs[:N_TEST_OGS]
            print(f"  Subsampled to {N_TEST_OGS} OGs")

        for og_idx, (row, groups, selection) in enumerate(
            tqdm(eligible_ogs, desc=f"Generating ({rank})")
        ):
            source_class, target_class, source_proteins, target_protein = selection
            target_name, target_seq, target_tid = target_protein

            # Build context from source class proteins
            context_seqs = [seq for _, seq, _ in source_proteins[:MAX_CONTEXT_PROTEINS]]
            source_seq = context_seqs[-1]  # last context protein = source baseline
            context_seqs += [source_seq[0]]

            # Tokenize context (everything except last token of last protein)
            context_input_ids = (
                tokenizer.tokenize_multi_proteins(
                    proteins=context_seqs,
                    flipped=False,
                    add_sep=False,
                    return_list=False,
                )[:-1]
                .unsqueeze(0)
                .to(DEVICE)
            )

            # Baseline: source vs target
            baseline_blosum = blosum_similarity(source_seq, target_seq)
            baseline_identity = sequence_identity(source_seq, target_seq)

            for layer_idx in STEER_LAYERS:
                if layer_idx not in all_probes.get(rank, {}):
                    continue
                probe = all_probes[rank][layer_idx]

                # Get class indices
                source_class_idx = probe["rank_to_classidx"].get(source_class, -1)
                target_class_idx = probe["rank_to_classidx"].get(target_class, -1)
                if source_class_idx == -1 or target_class_idx == -1:
                    continue

                # Compute steering directions
                steer_toward_target = compute_mean_steering_direction(
                    probe, target_class_idx, alpha=1.0  # alpha applied below
                )
                steer_toward_source = compute_mean_steering_direction(
                    probe, source_class_idx, alpha=1.0
                )

                for alpha in STEER_ALPHA:
                    for sample_idx in range(N_SAMPLES_PER_CONDITION):
                        sample_seed = SEED + og_idx * 1000 + sample_idx

                        # Generate: steer toward target
                        torch.manual_seed(sample_seed)
                        generated = {
                            "right": generate_with_steering(
                                model,
                                tokenizer,
                                context_input_ids,
                                layer_idx=layer_idx,
                                steering_vector=alpha * steer_toward_target,
                                max_len=len(source_seq) + 50,
                                eos_token_id=eos_token_id,
                            )
                        }

                        # Generate: no steering
                        if DO_GEN_NEUTRAL:
                            torch.manual_seed(sample_seed)
                            generated["neutral"] = generate_with_steering(
                                model,
                                tokenizer,
                                context_input_ids,
                                layer_idx=layer_idx,
                                steering_vector=None,
                                max_len=len(source_seq) + 50,
                                eos_token_id=eos_token_id,
                            )

                        # Generate: steer toward source (negative control)
                        if DO_GEN_WRONG:
                            torch.manual_seed(sample_seed)
                            generated["wrong"] = generate_with_steering(
                                model,
                                tokenizer,
                                context_input_ids,
                                layer_idx=layer_idx,
                                steering_vector=alpha * steer_toward_source,
                                max_len=len(source_seq) + 50,
                                eos_token_id=eos_token_id,
                            )

                        # Save every generated sequence with its generation metadata
                        for label, gen_seq in generated.items():
                            fasta_header = (
                                f"og={row['og']}|rank={rank}|layer={layer_idx}|alpha={alpha}"
                                f"|sample={sample_idx}|seed={sample_seed}|mode={label}"
                                f"|source_class={source_class}|target_class={target_class}"
                            )
                            write_fasta_record(fasta_handle, fasta_header, gen_seq)

                        # Score all generated sequences
                        scores = {}
                        for label, gen_seq in generated.items():
                            if len(gen_seq) < 10:
                                # Generation failed or too short
                                scores[label] = {
                                    "blosum_to_target": float("nan"),
                                    "identity_to_target": float("nan"),
                                    "blosum_to_source": float("nan"),
                                    "identity_to_source": float("nan"),
                                    "gen_len": len(gen_seq),
                                }
                                continue

                            scores[label] = {
                                "blosum_to_target": blosum_similarity(
                                    gen_seq, target_seq
                                ),
                                "identity_to_target": sequence_identity(
                                    gen_seq, target_seq
                                ),
                                "blosum_to_source": blosum_similarity(
                                    gen_seq, source_seq
                                ),
                                "identity_to_source": sequence_identity(
                                    gen_seq, source_seq
                                ),
                                "gen_len": len(gen_seq),
                            }
                        # breakpoint()
                        result = {  # type: ignore
                            "og": row["og"],
                            "rank": rank,
                            "layer": layer_idx,
                            "alpha": alpha,
                            "sample_idx": sample_idx,
                            "source_class": source_class,
                            "target_class": target_class,
                            "source_len": len(source_seq),
                            "target_len": len(target_seq),
                            "baseline_blosum": baseline_blosum,
                            "baseline_identity": baseline_identity,
                        }  # type: ignore
                        for label, label_scores in scores.items():
                            for metric, val in label_scores.items():
                                result[f"{label}_{metric}"] = val  # type: ignore

                        all_results.append(result)

                        metric_parts = [
                            f"right={scores['right']['blosum_to_target']:.3f}"
                        ]
                        if "neutral" in scores:
                            metric_parts.append(
                                f"neutral={scores['neutral']['blosum_to_target']:.3f}"
                            )
                        if "wrong" in scores:
                            metric_parts.append(
                                f"wrong={scores['wrong']['blosum_to_target']:.3f}"
                            )

                        print(
                            f"  {row['og']} L={layer_idx} α={alpha} s={sample_idx}: "
                            f"source class={source_class} target class={target_class} "
                            f"baseline_blosum={baseline_blosum:.3f} "
                            + " ".join(metric_parts)
                        )

            del context_input_ids
            torch.cuda.empty_cache()

            # Save periodically
            if (og_idx + 1) % 10 == 0:
                df_temp = pd.DataFrame(all_results)
                df_temp.to_csv(
                    OUTPUT_DIR / f"generation_results_{rank}_checkpoint.csv",
                    index=False,
                )

    fasta_handle.close()

    # ── Save final results ──
    df = pd.DataFrame(all_results)
    output_path = OUTPUT_DIR / f"generation_results_{rank}_seed{SEED}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Generated sequences saved to {fasta_path}")

    # ── Summary ──
    if len(df) > 0:
        print(f"\n{'═' * 70}")
        print("SUMMARY")
        print(f"{'═' * 70}")

        for rank in df["rank"].unique():
            print(f"\n  Rank: {rank}")
            rank_df = df[df["rank"] == rank]

            agg_kwargs = {
                "baseline": ("baseline_blosum", "mean"),
                "right": ("right_blosum_to_target", "mean"),
                "n": ("og", "count"),
            }
            if "neutral_blosum_to_target" in rank_df.columns:
                agg_kwargs["neutral"] = ("neutral_blosum_to_target", "mean")
            if "wrong_blosum_to_target" in rank_df.columns:
                agg_kwargs["wrong"] = ("wrong_blosum_to_target", "mean")

            summary = (
                rank_df.groupby(["layer", "alpha"]).agg(**agg_kwargs).reset_index()
            )

            if "neutral" in summary.columns:
                summary["delta_right_neutral"] = summary["right"] - summary["neutral"]
            if "wrong" in summary.columns:
                summary["delta_right_wrong"] = summary["right"] - summary["wrong"]

            header_parts = [
                f"{'Layer':>6}",
                f"{'Alpha':>8}",
                f"{'Baseline':>10}",
            ]
            row_parts = [
                lambda r: f"{r['layer']:>6}",
                lambda r: f"{r['alpha']:>8.3f}",
                lambda r: f"{r['baseline']:>10.3f}",
            ]

            if "neutral" in summary.columns:
                header_parts.append(f"{'Neutral':>10}")
                row_parts.append(lambda r: f"{r['neutral']:>10.3f}")

            header_parts.append(f"{'Right':>10}")
            row_parts.append(lambda r: f"{r['right']:>10.3f}")

            if "wrong" in summary.columns:
                header_parts.append(f"{'Wrong':>10}")
                row_parts.append(lambda r: f"{r['wrong']:>10.3f}")

            if "delta_right_neutral" in summary.columns:
                header_parts.append(f"{'Δ(R-N)':>10}")
                row_parts.append(lambda r: f"{r['delta_right_neutral']:>10.3f}")
            if "delta_right_wrong" in summary.columns:
                header_parts.append(f"{'Δ(R-W)':>10}")
                row_parts.append(lambda r: f"{r['delta_right_wrong']:>10.3f}")

            header_parts.append(f"{'N':>5}")
            row_parts.append(lambda r: f"{int(r['n']):>5}")

            header_line = "  " + " ".join(header_parts)
            print(header_line)
            print(f"  {'─' * (len(header_line) - 2)}")
            for _, r in summary.iterrows():
                print("  " + " ".join(part(r) for part in row_parts))


if __name__ == "__main__":
    main()_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run()
"""
Run dayhoff and add steering vector via forward hook at specified layer.
Check if PPL decreases when steered with right vs wrong taxon label.

Uses register_forward_hook to inject perturbation at any layer,
so the steered hidden state passes through all downstream layers,
layernorm, and lm_head via the model's own forward pass.
"""

import pickle
import torch
import torch.nn.functional as F
import random
import pandas as pd
from tqdm import tqdm
from deimm.model.tokenizer import PureARTokenizer
from deimm.utils.training_utils import (
    load_convert_parent,
    seed_everything,
)
from deimm.utils.constants import (
    MSA_PAD,
    RANK,
    PRETRAIN_DIR,
)
from pathlib import Path

# ═══════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════

TEST_FPATH = "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet"
RANKS_TO_PROBE = "class"
PROBE_PATH = Path("/scratch/suyuelyu/deimm/results/probe_taxon/")

# Layer → probe file mapping
# Adjust paths to where your pilot probes live
# LAYER_PROBE_MAP = {
#     8: PROBE_PATH / "pilot" / "probe_domain_layer8.pkl",
#     16: PROBE_PATH / "pilot" / "probe_domain_layer16.pkl",
#     24: PROBE_PATH / "pilot" / "probe_domain_layer24.pkl",
#     32: PROBE_PATH / "domain_ce_mmap" / "probe_domain_data.pkl",
# }
# STEER_LAYERS = [16, 24, 28, -1]  # -1 means last layer (32 for Dayhoff)
STEER_LAYERS = [16]
LAYER_PROBE_MAP = {
    16: PROBE_PATH
    / f"{RANKS_TO_PROBE}_ce_mmap_lyr16"
    / f"probe_{RANKS_TO_PROBE}_data.pkl",
}


# Layer indexing:
#   hidden_states[0] = embedding, hidden_states[k] = output of decoder layer k-1
#   "Layer 8" means hidden_states[8] = output of model.model.layers[7]
#   To steer at "layer 8", hook model.model.layers[7] (modifies its output)
#   Then perturbation flows through layers[8]...layers[31] → layernorm → lm_head
def hidden_idx_to_hook_layer(layer_idx: int) -> int:
    """Convert hidden_states index to model.model.layers index for hooking."""
    return layer_idx - 1 if layer_idx != -1 else 31


SEED = 3525
STEER_GRAD_ALPHA = [1.0, 5.0, 10.0]  # scaling factors for steering vector
N_TEST_PROTEINS = 50  # small subset for quick testing

seed_everything(SEED)
rand_generator = torch.Generator().manual_seed(SEED)


# ═══════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════


def load_dayhoff_model_tokenizer() -> tuple[torch.nn.Module, PureARTokenizer]:
    tokenizer_config = {
        "vocab_path": "vocab_UL_ALPHABET_PLUS.txt",
        "allow_unk": False,
    }
    tokenizer = PureARTokenizer(**tokenizer_config)
    pretrained_jamba_model = load_convert_parent(
        tokenizer(MSA_PAD),
        RANK,
        PRETRAIN_DIR,
        load_step=-1,
        evodiff2=True,
        tokenizer=None,
        new_vocab_size=None,
        use_flash_attention_2=True,
    )
    pretrained_jamba_model = pretrained_jamba_model.half().to("cuda")
    pretrained_jamba_model = pretrained_jamba_model.eval()
    print("Model loaded")
    return pretrained_jamba_model, tokenizer


# ═══════════════════════════════════════════════
# PROBE LOADING
# ═══════════════════════════════════════════════


def load_linear_probe(probe_path: str | Path) -> dict:
    print(f"Loading probe from {probe_path}...")
    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)
    return {
        "linear_weights": torch.from_numpy(probe_data["W"]).float().to("cuda"),
        "intercept": torch.from_numpy(probe_data["intercept"]).float().to("cuda"),
        "species_to_rank": probe_data["rank_mapping"],
        "rank_to_classidx": {lab: idx for idx, lab in enumerate(probe_data["classes"])},
        "n_classes": len(probe_data["classes"]),
    }


# ═══════════════════════════════════════════════
# STEERING HOOK
# ═══════════════════════════════════════════════


class SteeringHook:
    """
    Forward hook that adds a steering vector to the last N positions
    of the hidden state at a specified layer.

    Usage:
        hook = SteeringHook(steering_vector, last_protein_len)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        output = model(input_ids=input_ids, ...)
        handle.remove()
    """

    def __init__(self, steering_vector: torch.Tensor, last_protein_len: int):
        """
        Args:
            steering_vector: [seq_len, hidden_dim] perturbation to add
            last_protein_len: length of the last protein in the sequence
        """
        self.steering_vector = steering_vector
        self.last_protein_len = last_protein_len

    def __call__(self, module, input, output):
        # output structure depends on the layer type
        # For most transformer layers, output is a tuple where output[0] is hidden states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Add steering to last protein positions
        # Causal LM: hidden at position i produces logits predicting token i+1
        # So to affect the last protein (L tokens), steer positions -(L+1) to -2
        L = self.last_protein_len
        hidden[:, -(L + 1) : -1, :] += self.steering_vector.half().to(hidden.device)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


# ═══════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════


@torch.inference_mode()
def get_unsteered_output(
    model: torch.nn.Module,
    protein_names: list,
    lins: list,
    seqs: list,
    tokenizer: PureARTokenizer,
    n_max_protein: int = 64,
    og_seed: int = 0,
) -> dict | None:
    """Run forward pass, return hidden states at all steer layers + unsteered logits."""
    n_max_protein = min(n_max_protein, len(seqs))
    # Use deterministic seed per OG so Phase 2 can reproduce the same context
    og_gen = torch.Generator().manual_seed(og_seed)
    idxs = torch.randperm(len(seqs), generator=og_gen)[:n_max_protein]
    seqs = [seqs[i] for i in idxs]
    protein_names = [protein_names[i] for i in idxs]
    lins = [lins[i] for i in idxs]

    last_protein_len = len(seqs[-1])

    input_ids = (
        tokenizer.tokenize_multi_proteins(
            proteins=seqs, flipped=False, add_sep=False, return_list=False
        )[:-1]
        .unsqueeze(0)
        .to("cuda")
    )

    output = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    # Logits for the last protein (shifted by 1 for causal LM)
    logits = output.logits[0, -(last_protein_len + 1) : -1, :]

    # Extract hidden states at each steer layer for the last protein
    # Also shifted: hidden[i] produces logits[i] which predicts token[i+1]
    hidden_states = output.hidden_states
    layer_hiddens = {}
    for layer_idx in STEER_LAYERS:
        # hidden_states has n_layers + 1 entries (embedding + each layer)
        # layer_idx=0 is embedding, layer_idx=k is output of layer k-1
        # So for decoder layer k, we want hidden_states[k+1]
        # But check: if model outputs 33 hidden states for 32 layers,
        # hidden_states[0] = embedding, hidden_states[32] = last layer output
        h = hidden_states[layer_idx][0, -(last_protein_len + 1) : -1, :]
        layer_hiddens[layer_idx] = h.cpu()

    del output, hidden_states, input_ids
    torch.cuda.empty_cache()

    return {
        "species_id": lins[-1],
        "seq": seqs[-1],
        "seqs_used": seqs,  # save full context for rebuilding input_ids
        "protein_names": protein_names,
        "last_protein_len": last_protein_len,
        "layer_hiddens": layer_hiddens,
        "unsteered_logits": logits.cpu(),
    }


def compute_steering_vector(
    hidden: torch.Tensor,
    linear_weights: torch.Tensor,
    intercept: torch.Tensor,
    target_idx: int,
    alpha: float,
) -> torch.Tensor:
    """
    Compute gradient-based steering vector.

    Returns: [seq_len, hidden_dim] steering vector to ADD for steering toward target.
    """
    h = hidden.float().to("cuda")
    prob = torch.softmax(h @ linear_weights + intercept, dim=-1)  # [L, n_cls]
    e_target = torch.zeros(prob.shape[-1], device="cuda")
    e_target[target_idx] = 1.0
    grad = (prob - e_target) @ linear_weights.t()  # [L, hidden_dim]
    return -alpha * grad  # negative gradient = toward target


@torch.inference_mode()
def run_steered_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    last_protein_len: int,
    layer_idx: int,
) -> torch.Tensor:
    """Run forward pass with steering hook, return logits for last protein."""
    hook_layer = hidden_idx_to_hook_layer(layer_idx)
    hook = SteeringHook(steering_vector, last_protein_len)
    handle = model.model.layers[hook_layer].register_forward_hook(hook)

    try:
        output = model(input_ids=input_ids, return_dict=True)
        logits = output.logits[0, -(last_protein_len + 1) : -1, :]
    finally:
        handle.remove()

    return logits.cpu()


def calc_ppl(
    logits: torch.Tensor, target_seq: str, tokenizer: PureARTokenizer
) -> tuple[torch.Tensor, float]:
    """Calculate per-position and mean PPL."""
    target_ids = tokenizer.tokenize_protein(target_seq).to("cpu")
    logits = logits.to("cpu")
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        reduction="none",
    )
    per_pos_ppl = torch.exp(ce)
    mean_ppl = torch.exp(ce.mean()).item()  # correct: mean in log space first
    return per_pos_ppl, mean_ppl


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════


def main() -> None:
    meta_grouped_test = pd.read_parquet(TEST_FPATH)

    # Subsample for quick test
    if N_TEST_PROTEINS is not None and len(meta_grouped_test) > N_TEST_PROTEINS:
        meta_grouped_test = meta_grouped_test.sample(
            n=N_TEST_PROTEINS, random_state=SEED
        ).reset_index(drop=True)
        print(f"Subsampled to {N_TEST_PROTEINS} test groups")

    model, tokenizer = load_dayhoff_model_tokenizer()

    # Load probes for each layer
    layer_probes = {}
    for layer_idx in STEER_LAYERS:
        probe_path = LAYER_PROBE_MAP[layer_idx]
        if probe_path.exists():
            layer_probes[layer_idx] = load_linear_probe(probe_path)
        else:
            print(f"  WARNING: probe not found for layer {layer_idx}: {probe_path}")

    # ── Phase 1: Get unsteered outputs + hidden states ──
    print("Phase 1: Computing unsteered outputs...")
    conditions = {}
    for idx, row in tqdm(
        meta_grouped_test.iterrows(),
        total=len(meta_grouped_test),
        desc="Unsteered forward",
    ):
        result = get_unsteered_output(
            model=model,
            protein_names=row["protein"],
            lins=row["taxid"],
            seqs=row["seq"],
            tokenizer=tokenizer,
            og_seed=SEED + idx,
        )
        if result is None:
            continue

        _, unsteered_ppl = calc_ppl(
            result["unsteered_logits"], result["seq"], tokenizer
        )
        # Also store input_ids for reuse in steered forward passes
        result["unsteered_ppl"] = unsteered_ppl
        conditions[row["og"]] = result

    # ── Validation: confirm hook-based forward matches Phase 1 ──
    print("\nValidation: checking forward path consistency...")
    test_og = next(iter(conditions))
    test_cond = conditions[test_og]
    val_input_ids = (
        tokenizer.tokenize_multi_proteins(
            proteins=test_cond["seqs_used"],
            flipped=False,
            add_sep=False,
            return_list=False,
        )[:-1]
        .unsqueeze(0)
        .to("cuda")
    )
    # Zero steering at last layer = should reproduce original logits
    zero_steer = torch.zeros_like(test_cond["layer_hiddens"][STEER_LAYERS[-1]])
    val_logits = run_steered_forward(
        model,
        val_input_ids,
        zero_steer,
        test_cond["last_protein_len"],
        STEER_LAYERS[-1],
    )
    max_diff = (val_logits - test_cond["unsteered_logits"]).abs().max().item()
    _, val_ppl = calc_ppl(val_logits, test_cond["seq"], tokenizer)
    print(f"  Max logit diff: {max_diff:.6f}")
    print(f"  Unsteered PPL (Phase 1): {test_cond['unsteered_ppl']:.4f}")
    print(f"  Unsteered PPL (hook):    {val_ppl:.4f}")
    if max_diff > 0.1:
        print(
            "  WARNING: large discrepancy! Check layer indexing and position slicing."
        )
    else:
        print("  OK — forward paths match.")
    del val_input_ids, val_logits, zero_steer
    torch.cuda.empty_cache()

    # ── Phase 2: Steer at each layer × alpha × right/wrong ──
    print("\nPhase 2: Steering...")
    results = []

    for og, cond in tqdm(conditions.items(), desc="Steering"):
        species_id = cond["species_id"]
        seq = cond["seq"]
        last_protein_len = cond["last_protein_len"]

        # Rebuild input_ids from the exact same seqs used in Phase 1
        input_ids = (
            tokenizer.tokenize_multi_proteins(
                proteins=cond["seqs_used"],
                flipped=False,
                add_sep=False,
                return_list=False,
            )[:-1]
            .unsqueeze(0)
            .to("cuda")
        )

        for layer_idx in STEER_LAYERS:
            if layer_idx not in layer_probes:
                continue

            probe = layer_probes[layer_idx]
            hidden = cond["layer_hiddens"].get(layer_idx)
            if hidden is None:
                continue

            # Get right rank
            right_rank = probe["species_to_rank"].get(species_id, -1)
            right_rank_idx = probe["rank_to_classidx"].get(right_rank, -1)
            if right_rank_idx == -1:
                continue

            # Pick a random wrong rank
            wrong_rank = right_rank
            while wrong_rank == right_rank:
                wrong_rank = random.choice(list(probe["rank_to_classidx"].keys()))
            wrong_rank_idx = probe["rank_to_classidx"][wrong_rank]

            for alpha in STEER_GRAD_ALPHA:
                # Compute steering vectors
                right_steer = compute_steering_vector(
                    hidden,
                    probe["linear_weights"],
                    probe["intercept"],
                    right_rank_idx,
                    alpha,
                )
                wrong_steer = compute_steering_vector(
                    hidden,
                    probe["linear_weights"],
                    probe["intercept"],
                    wrong_rank_idx,
                    alpha,
                )

                # Run steered forward passes
                right_logits = run_steered_forward(
                    model,
                    input_ids,
                    right_steer,
                    last_protein_len,
                    layer_idx,
                )
                wrong_logits = run_steered_forward(
                    model,
                    input_ids,
                    wrong_steer,
                    last_protein_len,
                    layer_idx,
                )

                _, right_ppl = calc_ppl(right_logits, seq, tokenizer)
                _, wrong_ppl = calc_ppl(wrong_logits, seq, tokenizer)

                results.append(
                    {
                        "og": og,
                        "layer": layer_idx,
                        "alpha": alpha,
                        "right_rank": right_rank,
                        "wrong_rank": wrong_rank,
                        "unsteered_ppl": cond["unsteered_ppl"],
                        "right_ppl": right_ppl,
                        "wrong_ppl": wrong_ppl,
                    }
                )

                print(
                    f"  {og} layer={layer_idx} α={alpha}: "
                    f"unsteer={cond['unsteered_ppl']:.2f} "
                    f"right={right_ppl:.2f} wrong={wrong_ppl:.2f}"
                )

        del input_ids
        torch.cuda.empty_cache()

    # ── Save results ──
    output_dir = PROBE_PATH / "steer_ppl"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    output_path = (
        output_dir
        / f"{RANKS_TO_PROBE}_lyr_{'-'.join(map(str, STEER_LAYERS))}_steer_ppl.parquet"
    )
    df.to_parquet(output_path)
    print(f"\nResults saved to {output_path}")

    # ── Summary ──
    if len(df) > 0:
        print("\n" + "═" * 70)
        print("SUMMARY: Mean PPL by layer and alpha")
        print("═" * 70)
        summary = (
            df.groupby(["layer", "alpha"])
            .agg(
                unsteered=("unsteered_ppl", "mean"),
                right=("right_ppl", "mean"),
                wrong=("wrong_ppl", "mean"),
                n=("og", "count"),
            )
            .reset_index()
        )

        summary["delta"] = summary["wrong"] - summary["right"]

        print(
            f"{'Layer':>6} {'Alpha':>8} {'Unsteer':>10} {'Right':>10} "
            f"{'Wrong':>10} {'Δ(W-R)':>10} {'N':>5}"
        )
        print("─" * 60)
        for _, r in summary.iterrows():
            print(
                f"{r['layer']:>6} {r['alpha']:>8.3f} {r['unsteered']:>10.2f} "
                f"{r['right']:>10.2f} {r['wrong']:>10.2f} "
                f"{r['delta']:>10.2f} {int(r['n']):>5}"
            )


if __name__ == "__main__":
    main()

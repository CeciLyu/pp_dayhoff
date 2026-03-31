"""Hook-based PPL steering for hierarchical probe checkpoints.

This script loads a trained hierarchical probe run directory, restores probe
weights from an explicit checkpoint path, and measures PPL change when steering
at a chosen rank.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from deimm.utils.training_utils import seed_everything

from hier_probe_steer_utils import (
    LastProteinSteeringHook,
    build_probe,
    compute_adaptive_steering_vector,
    compute_fixed_steering_vector,
    hidden_idx_to_hook_layer,
    load_dayhoff_model_tokenizer,
    load_hier_probe_bundle,
    load_probe_state_dict_from_checkpoint,
    pick_wrong_group_idx,
    resolve_data_path,
    resolve_layer_idx,
    sample_context_from_og,
    species_cls_to_rank_group,
    species_tid_to_cls,
    validate_rank,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical rank steering via forward hook (PPL)")
    p.add_argument("--save_dir", type=str, required=True, help="Hierarchical probe run dir")
    p.add_argument("--ckpt_path", type=str, required=True, help="Explicit checkpoint path")
    p.add_argument("--rank", type=str, required=True, help="Rank to steer: species/genus/.../domain")

    p.add_argument("--layer_idx", type=int, default=None, help="Hidden-state index to steer; default from args.json")
    p.add_argument("--data_path", type=str, default=None, help="Parquet path; default args.json:test_parquet")

    p.add_argument("--n_max_protein", type=int, default=None, help="Max proteins per OG; default from args.json")
    p.add_argument("--n_test_ogs", type=int, default=0, help="Optional OG cap (0 means all)")
    p.add_argument("--alphas", type=str, default="5.0,10.0,20.0", help="Comma-separated steering strengths")

    p.add_argument("--steer_mode", type=str, default="fixed", choices=["fixed", "adaptive"], help="Steering vector mode")
    p.add_argument("--include_wrong", action="store_true", help="Also run wrong-rank control")

    p.add_argument("--seed", type=int, default=3525)
    p.add_argument("--output_subdir", type=str, default="steer_ppl_hier")
    return p.parse_args()


def parse_float_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("--alphas must contain at least one numeric value")
    return vals


def calc_ppl(logits: torch.Tensor, target_seq: str, tokenizer) -> float:
    target_ids = tokenizer.tokenize_protein(target_seq).to(logits.device)
    ce = F.cross_entropy(logits, target_ids, reduction="none")
    return float(torch.exp(ce.mean()).item())


@torch.inference_mode()
def run_forward_logits(model: torch.nn.Module, input_ids: torch.Tensor, last_len: int) -> torch.Tensor:
    out = model(input_ids=input_ids, return_dict=True)
    return out.logits[0, -(last_len + 1) : -1, :]


@torch.inference_mode()
def get_unsteered_hidden_and_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_idx: int,
    last_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[layer_idx][0, -(last_len + 1) : -1, :]
    logits = out.logits[0, -(last_len + 1) : -1, :]
    return hidden, logits


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.alphas)

    seed_everything(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    ckpt_path = Path(args.ckpt_path)

    bundle = load_hier_probe_bundle(save_dir, device=device)
    validate_rank(args.rank, bundle)

    layer_idx = resolve_layer_idx(bundle.layer_idx, args.layer_idx)
    data_path = resolve_data_path(bundle.args, args.data_path)
    n_max_protein = int(
        bundle.args.get("n_max_protein", 64)
        if args.n_max_protein is None
        else args.n_max_protein
    )

    model, tokenizer = load_dayhoff_model_tokenizer(device)

    state_dict = load_probe_state_dict_from_checkpoint(ckpt_path)
    probe = build_probe(bundle.hidden_dim, bundle.n_species, state_dict, device=device)

    n_layers = len(model.model.layers)
    hook_layer = hidden_idx_to_hook_layer(layer_idx, n_layers)
    print(f"Steering rank={args.rank} at hidden layer_idx={layer_idx} (hook layer={hook_layer})")

    df = pd.read_parquet(data_path)
    if args.n_test_ogs > 0 and len(df) > args.n_test_ogs:
        df = df.sample(n=args.n_test_ogs, random_state=args.seed).reset_index(drop=True)

    out_dir = save_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    rng = random.Random(args.seed)

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="PPL steering"):
        seqs = row["seq"]
        taxids = row["taxid"]
        if not seqs or not taxids:
            continue

        sample_gen = torch.Generator().manual_seed(args.seed + int(row_idx))
        sampled_seqs, _, last_len, last_taxid = sample_context_from_og(
            seqs=seqs,
            taxids=taxids,
            n_max_protein=n_max_protein,
            generator=sample_gen,
        )

        species_cls = species_tid_to_cls(last_taxid, bundle)
        if species_cls is None:
            continue

        right_idx = species_cls_to_rank_group(species_cls, args.rank, bundle)
        if right_idx is None:
            continue

        input_ids = (
            tokenizer.tokenize_multi_proteins(
                proteins=sampled_seqs, flipped=False, add_sep=False, return_list=False
            )[:-1]
            .unsqueeze(0)
            .to(device)
        )

        hidden, unsteered_logits = get_unsteered_hidden_and_logits(
            model=model,
            input_ids=input_ids,
            layer_idx=layer_idx,
            last_len=last_len,
        )
        unsteered_ppl = calc_ppl(unsteered_logits, sampled_seqs[-1], tokenizer)

        for alpha in alphas:
            if args.steer_mode == "fixed":
                right_steer = compute_fixed_steering_vector(
                    probe=probe,
                    rank=args.rank,
                    target_group_idx=right_idx,
                    alpha=alpha,
                    bundle=bundle,
                    device=device,
                )
            else:
                right_steer = compute_adaptive_steering_vector(
                    hidden=hidden,
                    probe=probe,
                    rank=args.rank,
                    target_group_idx=right_idx,
                    alpha=alpha,
                    bundle=bundle,
                )

            right_hook = LastProteinSteeringHook(right_steer, last_len)
            handle = model.model.layers[hook_layer].register_forward_hook(right_hook)
            try:
                right_logits = run_forward_logits(model, input_ids, last_len)
            finally:
                handle.remove()
            right_ppl = calc_ppl(right_logits, sampled_seqs[-1], tokenizer)

            entry: dict[str, float | int | str] = {
                "og": str(row["og"]),
                "rank": args.rank,
                "species_tid": int(last_taxid),
                "species_cls": int(species_cls),
                "right_group_idx": int(right_idx),
                "layer_idx": int(layer_idx),
                "alpha": float(alpha),
                "steer_mode": args.steer_mode,
                "unsteered_ppl": float(unsteered_ppl),
                "right_ppl": float(right_ppl),
                "delta_right_minus_unsteered": float(right_ppl - unsteered_ppl),
            }

            if args.include_wrong:
                wrong_idx = pick_wrong_group_idx(
                    right_group_idx=right_idx,
                    rank=args.rank,
                    bundle=bundle,
                    rng=rng,
                )
                if args.steer_mode == "fixed":
                    wrong_steer = compute_fixed_steering_vector(
                        probe=probe,
                        rank=args.rank,
                        target_group_idx=wrong_idx,
                        alpha=alpha,
                        bundle=bundle,
                        device=device,
                    )
                else:
                    wrong_steer = compute_adaptive_steering_vector(
                        hidden=hidden,
                        probe=probe,
                        rank=args.rank,
                        target_group_idx=wrong_idx,
                        alpha=alpha,
                        bundle=bundle,
                    )

                wrong_hook = LastProteinSteeringHook(wrong_steer, last_len)
                handle = model.model.layers[hook_layer].register_forward_hook(wrong_hook)
                try:
                    wrong_logits = run_forward_logits(model, input_ids, last_len)
                finally:
                    handle.remove()
                wrong_ppl = calc_ppl(wrong_logits, sampled_seqs[-1], tokenizer)

                entry["wrong_group_idx"] = int(wrong_idx)
                entry["wrong_ppl"] = float(wrong_ppl)
                entry["delta_wrong_minus_unsteered"] = float(wrong_ppl - unsteered_ppl)
                entry["delta_wrong_minus_right"] = float(wrong_ppl - right_ppl)

            rows.append(entry)

    if not rows:
        raise RuntimeError("No valid OG samples processed. Check rank/species overlap and inputs.")

    out_df = pd.DataFrame(rows)
    stem = f"ppl_rank-{args.rank}_layer-{layer_idx}_mode-{args.steer_mode}"
    out_csv = out_dir / f"{stem}.csv"
    out_parquet = out_dir / f"{stem}.parquet"
    out_df.to_csv(out_csv, index=False)
    out_df.to_parquet(out_parquet)

    summary = (
        out_df.groupby(["rank", "layer_idx", "alpha", "steer_mode"]).agg(
            n=("og", "count"),
            unsteered=("unsteered_ppl", "mean"),
            right=("right_ppl", "mean"),
            delta_right=("delta_right_minus_unsteered", "mean"),
        )
    ).reset_index()
    if args.include_wrong and "wrong_ppl" in out_df.columns:
        wrong_summary = (
            out_df.groupby(["rank", "layer_idx", "alpha", "steer_mode"]).agg(
                wrong=("wrong_ppl", "mean"),
                delta_wrong=("delta_wrong_minus_unsteered", "mean"),
                delta_wrong_minus_right=("delta_wrong_minus_right", "mean"),
            )
        ).reset_index()
        summary = summary.merge(
            wrong_summary,
            on=["rank", "layer_idx", "alpha", "steer_mode"],
            how="left",
        )

    summary_csv = out_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Saved detailed results: {out_csv}")
    print(f"Saved parquet results: {out_parquet}")
    print(f"Saved summary: {summary_csv}")


if __name__ == "__main__":
    main()

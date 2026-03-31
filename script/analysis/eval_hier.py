"""Per-position probe evaluation for hierarchical probe checkpoints.

This script mirrors eval.py behavior (context sampling per OG, probing last
protein per position), but uses the hierarchical species probe + rank
aggregation matrices saved by probe_taxon_online.py.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from tqdm import tqdm

from deimm.utils.constants import MSA_PAD
from deimm.utils.training_utils import seed_everything

from hier_probe_steer_utils import (
    build_probe,
    hidden_idx_to_hook_layer,
    load_dayhoff_model_tokenizer,
    load_hier_probe_bundle,
    load_probe_state_dict_from_checkpoint,
    resolve_data_path,
    resolve_layer_idx,
    validate_rank,
)

RANK_PRIORITY = ["species", "genus", "family", "order", "class", "phylum", "domain"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical probe per-position evaluation")
    p.add_argument("--save_dir", type=str, required=True, help="Hierarchical probe run dir")
    p.add_argument("--ckpt_path", type=str, required=True, help="Explicit checkpoint path")
    p.add_argument(
        "--rank",
        type=str,
        default="all",
        help="Rank(s) to evaluate: 'all' or comma-separated list",
    )

    p.add_argument(
        "--layer_idx",
        type=int,
        default=None,
        help="Hidden-state index to evaluate; default from args.json",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Parquet path; default args.json:test_parquet",
    )
    p.add_argument(
        "--taxonomy_mapping_file",
        type=str,
        default=None,
        help="Override taxonomy mapping file; default args.json:taxonomy_mapping_file",
    )

    p.add_argument("--seed", type=int, default=3525)
    p.add_argument("--n_samples_per_og", type=int, default=10)
    p.add_argument("--max_context_proteins", type=int, default=64)
    p.add_argument("--max_condition_pairs_per_og", type=int, default=1000)
    p.add_argument("--max_total_samples_per_og", type=int, default=3000)

    p.add_argument("--max_forward_batch_samples", type=int, default=1000)
    p.add_argument("--max_forward_batch_tokens", type=int, default=200000)
    p.add_argument("--save_every_og", type=int, default=5)

    p.add_argument("--output_subdir", type=str, default="per_pos_eval_hier")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing result files and start fresh",
    )
    return p.parse_args()


def load_taxonomy_mapping(mapping_file: str) -> dict[int, dict[str, int]]:
    with open(mapping_file, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected taxonomy mapping format: {mapping_file}")
    return obj


def resolve_eval_ranks(rank_arg: str, available_rank_data: list[str]) -> list[str]:
    available = set(["species", *available_rank_data])
    if rank_arg.strip().lower() == "all":
        return [r for r in RANK_PRIORITY if r in available]

    requested = [r.strip() for r in rank_arg.split(",") if r.strip()]
    if not requested:
        raise ValueError("--rank must not be empty")
    for r in requested:
        if r not in available:
            raise ValueError(
                f"Requested rank '{r}' not available. Available: {sorted(available)}"
            )
    return requested


def get_true_rank_ids_for_taxids(
    taxids: list[int],
    rank: str,
    taxid_to_std_ranks: dict[int, dict[str, int]],
) -> list[int | None]:
    if rank == "species":
        return [int(t) for t in taxids]

    out: list[int | None] = []
    for tid in taxids:
        rank_dict = taxid_to_std_ranks.get(int(tid), {})
        out.append(int(rank_dict[rank]) if rank in rank_dict else None)
    return out


def compute_sampling_budget(
    n_classes: int,
    n_samples_per_og: int,
    max_condition_pairs_per_og: int | None,
    max_total_samples_per_og: int | None,
) -> tuple[int, int]:
    raw_condition_pairs = 1 if n_classes < 2 else n_classes * n_classes
    condition_pairs_used = raw_condition_pairs
    if (
        max_condition_pairs_per_og is not None
        and condition_pairs_used > max_condition_pairs_per_og
    ):
        condition_pairs_used = max_condition_pairs_per_og

    samples_per_condition = n_samples_per_og
    if max_total_samples_per_og is not None:
        samples_per_condition = min(
            n_samples_per_og,
            max(1, max_total_samples_per_og // max(1, condition_pairs_used)),
        )
    return condition_pairs_used, samples_per_condition


def sample_proteins_for_og(
    rank_ids: list[int | None],
    max_context_proteins: int,
    src_rank_id: int | None = None,
    tgt_rank_id: int | None = None,
    rank_to_indices: dict[int | None, list[int]] | None = None,
) -> list[int] | None:
    if rank_to_indices is None:
        rank_to_indices = defaultdict(list)
        for i, rid in enumerate(rank_ids):
            rank_to_indices[rid].append(i)

    rank_idxs = list(range(len(rank_ids)))
    if len(rank_idxs) == 0:
        return None

    # Single-protein mode: choose exactly one index.
    # Prefer target class when provided so conditioning still makes sense.
    if max_context_proteins <= 1:
        if tgt_rank_id is not None:
            candidates = rank_to_indices.get(tgt_rank_id, [])
        elif src_rank_id is not None:
            candidates = rank_to_indices.get(src_rank_id, [])
        else:
            candidates = rank_idxs
        if len(candidates) == 0:
            return None
        return [random.choice(candidates)]

    if src_rank_id is not None and tgt_rank_id is not None:
        if src_rank_id != tgt_rank_id:
            src_idxs = rank_to_indices.get(src_rank_id, [])
            tgt_idxs = rank_to_indices.get(tgt_rank_id, [])
            if len(src_idxs) == 0 or len(tgt_idxs) == 0:
                return None

            # We already know max_context_proteins > 1 here.
            max_src = min(max_context_proteins - 1, len(src_idxs))
            if max_src < 1:
                return None

            n_src = random.randint(1, max_src)
            src_sample = random.sample(src_idxs, n_src)
            tgt_sample = random.sample(tgt_idxs, 1)
            return src_sample + tgt_sample

        rank_idxs = rank_to_indices.get(src_rank_id, [])
        if len(rank_idxs) == 0:
            return None

    # General sampling (>=2 proteins allowed)
    max_n = min(max_context_proteins, len(rank_idxs))
    if max_n < 2:
        return None
    n_proteins = random.randint(2, max_n)
    return random.sample(rank_idxs, n_proteins)

def sample_context_for_og(
    protein_names: list[Any],
    rank_ids: list[int | None],
    seqs: list[str],
    tokenizer,
    max_context_proteins: int,
    src_rank_id: int | None = None,
    tgt_rank_id: int | None = None,
    rank_to_indices: dict[int | None, list[int]] | None = None,
) -> dict[str, Any] | None:
    idxs = sample_proteins_for_og(
        rank_ids=rank_ids,
        max_context_proteins=max_context_proteins,
        src_rank_id=src_rank_id,
        tgt_rank_id=tgt_rank_id,
        rank_to_indices=rank_to_indices,
    )
    if idxs is None:
        return None

    seqs_sampled = [seqs[i] for i in idxs]
    input_ids = tokenizer.tokenize_multi_proteins(
        proteins=seqs_sampled,
        flipped=False,
        add_sep=False,
        return_list=False,
    )[:-1]

    return {
        "protein_names": [protein_names[i] for i in idxs],
        "rank_ids": [rank_ids[i] for i in idxs],
        "input_ids": input_ids,
        "input_len": int(input_ids.shape[0]),
        "last_protein_len": len(seqs_sampled[-1]),
    }


def build_padded_batch(
    sample_contexts: list[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sample_contexts)
    max_len = max(s["input_len"] for s in sample_contexts)

    input_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, sample in enumerate(sample_contexts):
        seq_len = sample["input_len"]
        ids = sample["input_ids"].to(device)
        input_ids[i, :seq_len] = ids
        attention_mask[i, :seq_len] = 1

    return input_ids, attention_mask


def iter_context_chunks(
    sample_contexts: list[dict[str, Any]],
    max_batch_samples: int,
    max_batch_tokens: int | None,
) -> Iterator[list[dict[str, Any]]]:
    if len(sample_contexts) == 0:
        return

    sorted_contexts = sorted(sample_contexts, key=lambda x: x["input_len"])
    chunk: list[dict[str, Any]] = []
    chunk_max_len = 0

    for sample in sorted_contexts:
        sample_len = sample["input_len"]
        next_max_len = max(chunk_max_len, sample_len)
        next_size = len(chunk) + 1
        exceeds_samples = next_size > max_batch_samples
        exceeds_tokens = max_batch_tokens is not None and (
            next_max_len * next_size > max_batch_tokens
        )

        if chunk and (exceeds_samples or exceeds_tokens):
            yield chunk
            chunk = [sample]
            chunk_max_len = sample_len
        else:
            chunk.append(sample)
            chunk_max_len = next_max_len

    if chunk:
        yield chunk


@torch.inference_mode()
def extract_last_protein_hidden_for_contexts(
    model: torch.nn.Module,
    sample_contexts: list[dict[str, Any]],
    pad_token_id: int,
    layer_idx: int,
    device: torch.device,
) -> list[torch.Tensor]:
    if not sample_contexts:
        return []

    input_ids, attention_mask = build_padded_batch(sample_contexts, pad_token_id, device)
    backbone = model.model if hasattr(model, "model") else model

    if not hasattr(backbone, "layers"):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_batch = out.hidden_states[layer_idx]
    elif layer_idx == -1:
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_batch = out.last_hidden_state
    else:
        n_model_layers = len(backbone.layers)
        hook_layer = hidden_idx_to_hook_layer(layer_idx, n_model_layers)
        captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["hidden"] = hidden

        handle = backbone.layers[hook_layer].register_forward_hook(_hook)
        try:
            backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        if "hidden" not in captured:
            raise RuntimeError(f"Failed to capture hidden states for layer_idx={layer_idx}")
        hidden_batch = captured["hidden"]

    outputs: list[torch.Tensor] = []
    for bidx, sample in enumerate(sample_contexts):
        seq_len = sample["input_len"]
        last_len = sample["last_protein_len"]
        outputs.append(hidden_batch[bidx, seq_len - last_len : seq_len, :])
    return outputs


@torch.inference_mode()
def predict_rank_ids_from_last_hidden(
    probe: torch.nn.Linear,
    last_hidden: torch.Tensor,
    rank: str,
    bundle,
) -> list[int]:
    logits = probe(last_hidden.float())
    if rank == "species":
        pred_cls = torch.argmax(logits, dim=-1)
        return [int(bundle.cls_to_tid[int(idx)]) for idx in pred_cls.tolist()]

    rank_data = bundle.rank_data[rank]
    probs = torch.softmax(logits, dim=-1)
    group_probs = probs @ rank_data.M.to(probs.device).T
    pred_group_idx = torch.argmax(group_probs, dim=-1)
    classes = rank_data.group_classes
    return [int(classes[int(i)]) for i in pred_group_idx.tolist()]


def get_result_path(
    out_dir: Path,
    rank: str,
    layer_idx: int,
    ckpt_path: Path,
) -> Path:
    ckpt_tag = ckpt_path.stem.replace(" ", "_")
    return out_dir / f"probe_taxon_per_pos_test_results_{rank}_lyr{layer_idx}_{ckpt_tag}.pkl"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    ckpt_path = Path(args.ckpt_path)

    bundle = load_hier_probe_bundle(save_dir, device=device)
    eval_ranks = resolve_eval_ranks(args.rank, list(bundle.rank_data.keys()))
    for r in eval_ranks:
        validate_rank(r, bundle)

    layer_idx = resolve_layer_idx(bundle.layer_idx, args.layer_idx)
    data_path = resolve_data_path(bundle.args, args.data_path)
    taxonomy_mapping_file = args.taxonomy_mapping_file or bundle.args.get(
        "taxonomy_mapping_file"
    )
    if not taxonomy_mapping_file:
        raise RuntimeError(
            "No taxonomy mapping file found. Provide --taxonomy_mapping_file or ensure args.json has taxonomy_mapping_file."
        )

    model, tokenizer = load_dayhoff_model_tokenizer(device)
    pad_token_id = int(tokenizer(MSA_PAD))

    state_dict = load_probe_state_dict_from_checkpoint(ckpt_path)
    probe = build_probe(bundle.hidden_dim, bundle.n_species, state_dict, device=device)

    meta_grouped_test = pd.read_parquet(data_path)
    taxid_to_std_ranks = load_taxonomy_mapping(str(taxonomy_mapping_file))

    out_dir = save_dir / args.output_subdir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded test data with {len(meta_grouped_test)} orthologous groups")
    print(f"Evaluating ranks: {eval_ranks}")
    print(f"Using layer_idx={layer_idx}, ckpt={ckpt_path}")

    for rank in eval_ranks:
        output_path = get_result_path(out_dir, rank, layer_idx, ckpt_path)
        rank_results: list[dict[str, Any]] = []

        if output_path.exists() and not args.overwrite:
            print(f"Resuming existing results at {output_path}...")
            with open(output_path, "rb") as f:
                loaded = pickle.load(f)
            rank_results = [r for r in loaded if r.get("rank") == rank]

        ogs_evaluated = {r["og"] for r in rank_results}
        pending = len(meta_grouped_test) - len(ogs_evaluated)
        if pending <= 0:
            print(f"All OGs already evaluated for rank={rank}, skipping.")
            continue

        print(
            f"Evaluating rank={rank}. Existing OGs={len(ogs_evaluated)}, pending={pending}."
        )

        n_new_og_results = 0
        for row in tqdm(
            meta_grouped_test.itertuples(index=False),
            total=len(meta_grouped_test),
            desc=f"Eval {rank}",
        ):
            if row.og in ogs_evaluated:
                continue

            taxids = [int(t) for t in row.taxid]
            rank_ids = get_true_rank_ids_for_taxids(
                taxids=taxids,
                rank=rank,
                taxid_to_std_ranks=taxid_to_std_ranks,
            )

            rank_to_indices: dict[int | None, list[int]] = defaultdict(list)
            for idx, rid in enumerate(rank_ids):
                rank_to_indices[rid].append(idx)

            unique_rank_ids = set(r for r in rank_ids if r is not None)
            n_classes = len(unique_rank_ids)
            n_pairs_used, samples_per_condition = compute_sampling_budget(
                n_classes=n_classes,
                n_samples_per_og=args.n_samples_per_og,
                max_condition_pairs_per_og=args.max_condition_pairs_per_og,
                max_total_samples_per_og=args.max_total_samples_per_og,
            )

            condition_pairs: list[tuple[int | None, int | None]]
            if n_classes < 2:
                condition_pairs = [(None, None)]
            else:
                condition_pairs = []
                unique_rank_ids_list = [int(x) for x in unique_rank_ids]
                for src in unique_rank_ids_list:
                    for tgt in unique_rank_ids_list:
                        condition_pairs.append((src, tgt))
                if len(condition_pairs) > n_pairs_used:
                    random.shuffle(condition_pairs)
                    condition_pairs = condition_pairs[:n_pairs_used]

            all_sample_contexts: list[dict[str, Any]] = []
            for src_rank_id, tgt_rank_id in condition_pairs:
                for _ in range(samples_per_condition):
                    sample_ctx = sample_context_for_og(
                        protein_names=row.protein,
                        rank_ids=rank_ids,
                        seqs=row.seq,
                        tokenizer=tokenizer,
                        max_context_proteins=args.max_context_proteins,
                        src_rank_id=src_rank_id,
                        tgt_rank_id=tgt_rank_id,
                        rank_to_indices=rank_to_indices,
                    )
                    if sample_ctx is None:
                        break
                    sample_ctx["src_rank_id"] = src_rank_id
                    sample_ctx["tgt_rank_id"] = tgt_rank_id
                    all_sample_contexts.append(sample_ctx)

            for sample_chunk in iter_context_chunks(
                all_sample_contexts,
                max_batch_samples=args.max_forward_batch_samples,
                max_batch_tokens=args.max_forward_batch_tokens,
            ):
                chunk_hiddens = extract_last_protein_hidden_for_contexts(
                    model=model,
                    sample_contexts=sample_chunk,
                    pad_token_id=pad_token_id,
                    layer_idx=layer_idx,
                    device=device,
                )

                for sample_item, last_hidden in zip(sample_chunk, chunk_hiddens):
                    if sample_item is None:
                        continue
                    sample = sample_item
                    pred_tgt_rank_id = predict_rank_ids_from_last_hidden(
                        probe=probe,
                        last_hidden=last_hidden,
                        rank=rank,
                        bundle=bundle,
                    )
                    rank_results.append(
                        {
                            "og": row.og,
                            "rank": rank,
                            "protein_names": sample["protein_names"],
                            "rank_ids": sample["rank_ids"],
                            "layer": layer_idx,
                            "src_rank_id": sample["src_rank_id"],
                            "tgt_rank_id": sample["tgt_rank_id"],
                            "pred_tgt_rank_id": pred_tgt_rank_id,
                        }
                    )

            n_new_og_results += 1
            if args.save_every_og > 0 and n_new_og_results % args.save_every_og == 0:
                with open(output_path, "wb") as f:
                    pickle.dump(rank_results, f)

        with open(output_path, "wb") as f:
            pickle.dump(rank_results, f)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

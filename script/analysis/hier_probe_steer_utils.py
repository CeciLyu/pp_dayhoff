"""Shared utilities for hierarchical probe steering (PPL and generation).

This module centralizes loading trained hierarchical probe artifacts from a run
save directory, rebuilding the probe from a selected checkpoint, and helper
functions for rank-aware steering vectors.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from deimm.model.tokenizer import PureARTokenizer
from deimm.utils.constants import MSA_PAD, PRETRAIN_DIR, RANK
from deimm.utils.training_utils import load_convert_parent

TOKENIZER_CONFIG = {"vocab_path": "vocab_UL_ALPHABET_PLUS.txt", "allow_unk": False}
VALID_RANKS = ["species", "genus", "family", "order", "class", "phylum", "domain"]


@dataclass
class RankData:
    """Rank-specific artifacts used for steering."""

    M: torch.Tensor
    species_to_group: torch.Tensor
    group_classes: list[int]
    group_to_idx: dict[int, int]


@dataclass
class HierProbeBundle:
    """Loaded hierarchical probe run artifacts and metadata."""

    save_dir: Path
    args: dict[str, Any]
    artifacts: dict[str, Any]
    tid_to_cls: dict[int, int]
    cls_to_tid: dict[int, int]
    species_classes: list[int]
    rank_data: dict[str, RankData]
    layer_idx: int
    hidden_dim: int
    n_species: int


def load_run_args(save_dir: Path) -> dict[str, Any]:
    args_path = save_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json under save dir: {args_path}")
    with open(args_path, "r") as f:
        args = json.load(f)
    if not isinstance(args, dict):
        raise RuntimeError(f"args.json must contain a JSON object: {args_path}")
    return args


def _to_int_key_dict(d: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in d.items()}


def load_prediction_artifacts(save_dir: Path) -> dict[str, Any]:
    path = save_dir / "prediction_artifacts.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction artifacts file: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected dict in prediction artifacts: {path}")
    return obj


def load_hier_probe_bundle(save_dir: str | Path, device: torch.device) -> HierProbeBundle:
    save_dir = Path(save_dir)
    args = load_run_args(save_dir)
    artifacts = load_prediction_artifacts(save_dir)

    tid_to_cls = _to_int_key_dict(cast_dict(artifacts.get("tid_to_cls", {}), "tid_to_cls"))
    cls_to_tid = _to_int_key_dict(cast_dict(artifacts.get("cls_to_tid", {}), "cls_to_tid"))
    species_classes = [int(x) for x in artifacts.get("species_classes", [])]
    if not tid_to_cls or not cls_to_tid or not species_classes:
        raise RuntimeError("Invalid prediction_artifacts.pt: missing species mappings")

    rank_data: dict[str, RankData] = {}
    ranks_obj = cast_dict(artifacts.get("ranks", {}), "ranks")
    for rank_name, rank_payload in ranks_obj.items():
        payload = cast_dict(rank_payload, f"ranks.{rank_name}")
        M = payload.get("M")
        species_to_group = payload.get("species_to_group")
        group_classes_raw = payload.get("group_classes")
        if M is None or species_to_group is None or group_classes_raw is None:
            continue

        M_t = torch.as_tensor(M, dtype=torch.float32, device=device)
        species_to_group_t = torch.as_tensor(
            species_to_group, dtype=torch.long, device=device
        )
        group_classes = [int(x) for x in group_classes_raw]
        group_to_idx = {int(tid): int(i) for i, tid in enumerate(group_classes)}
        rank_data[str(rank_name)] = RankData(
            M=M_t,
            species_to_group=species_to_group_t,
            group_classes=group_classes,
            group_to_idx=group_to_idx,
        )

    layer_idx = int(args.get("layer_idx", artifacts.get("layer_idx", -1)))
    hidden_dim = int(artifacts.get("hidden_dim", 1280))
    n_species = int(artifacts.get("n_species", len(species_classes)))

    return HierProbeBundle(
        save_dir=save_dir,
        args=args,
        artifacts=artifacts,
        tid_to_cls=tid_to_cls,
        cls_to_tid=cls_to_tid,
        species_classes=species_classes,
        rank_data=rank_data,
        layer_idx=layer_idx,
        hidden_dim=hidden_dim,
        n_species=n_species,
    )


def cast_dict(obj: Any, name: str) -> dict[Any, Any]:
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected dict for {name}, got {type(obj).__name__}")
    return obj


def load_probe_state_dict_from_checkpoint(ckpt_path: str | Path) -> dict[str, torch.Tensor]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    if "model_state_dict" in obj:
        state = obj["model_state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid model_state_dict in checkpoint: {ckpt_path}")
        return state

    # Also accept direct nn.Linear state_dict (probe_hierarchical_model.pt style)
    if "weight" in obj and "bias" in obj:
        return obj

    raise RuntimeError(
        f"Could not find probe state_dict in checkpoint: {ckpt_path}. "
        "Expected 'model_state_dict' or direct {'weight','bias'}."
    )


def build_probe(
    hidden_dim: int,
    n_species: int,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> nn.Linear:
    probe = nn.Linear(hidden_dim, n_species, bias=True).to(device)
    probe.load_state_dict(state_dict)
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False
    return probe


def load_dayhoff_model_tokenizer(device: torch.device) -> tuple[nn.Module, PureARTokenizer]:
    tokenizer = PureARTokenizer(**TOKENIZER_CONFIG)
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
    model = model.half().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def resolve_data_path(args_dict: dict[str, Any], data_path_override: str | None) -> str:
    if data_path_override is not None:
        return data_path_override
    data_path = args_dict.get("test_parquet")
    if not data_path:
        raise RuntimeError(
            "No --data_path provided and args.json has no 'test_parquet' value"
        )
    return str(data_path)


def resolve_layer_idx(default_layer_idx: int, override_layer_idx: int | None) -> int:
    return default_layer_idx if override_layer_idx is None else int(override_layer_idx)


def hidden_idx_to_hook_layer(layer_idx: int, n_model_layers: int) -> int:
    if layer_idx == -1:
        return n_model_layers - 1
    if layer_idx <= 0 or layer_idx > n_model_layers:
        raise ValueError(
            f"Unsupported hidden state index {layer_idx}; expected 1..{n_model_layers} or -1"
        )
    return layer_idx - 1


def validate_rank(rank: str, bundle: HierProbeBundle) -> None:
    if rank not in VALID_RANKS:
        raise ValueError(f"Unsupported rank '{rank}'. Valid options: {VALID_RANKS}")
    if rank == "species":
        return
    if rank not in bundle.rank_data:
        available = sorted(bundle.rank_data.keys())
        raise ValueError(
            f"Rank '{rank}' not found in prediction artifacts. Available ranks: {available}"
        )


def species_tid_to_cls(species_tid: int, bundle: HierProbeBundle) -> int | None:
    return bundle.tid_to_cls.get(int(species_tid))


def species_cls_to_rank_group(cls_idx: int, rank: str, bundle: HierProbeBundle) -> int | None:
    if rank == "species":
        return int(cls_idx)
    rank_data = bundle.rank_data[rank]
    group_idx = int(rank_data.species_to_group[int(cls_idx)].item())
    if group_idx < 0:
        return None
    return group_idx


def get_species_distribution_for_target(rank: str, target_group_idx: int, bundle: HierProbeBundle) -> torch.Tensor:
    """Return target species distribution over probe outputs (n_species,)."""
    n_species = bundle.n_species
    dist = torch.zeros((n_species,), dtype=torch.float32)

    if rank == "species":
        if target_group_idx < 0 or target_group_idx >= n_species:
            raise ValueError(
                f"Species target idx out of range: {target_group_idx} not in [0, {n_species})"
            )
        dist[target_group_idx] = 1.0
        return dist

    rank_data = bundle.rank_data[rank]
    if target_group_idx < 0 or target_group_idx >= rank_data.M.shape[0]:
        raise ValueError(
            f"Rank target idx out of range for rank={rank}: {target_group_idx}"
        )

    row = rank_data.M[target_group_idx].detach().cpu().float()
    s = float(row.sum().item())
    if s <= 0.0:
        raise RuntimeError(
            f"Rank={rank} target_group_idx={target_group_idx} has zero species membership"
        )
    return row / s


def compute_fixed_steering_vector(
    probe: nn.Linear,
    rank: str,
    target_group_idx: int,
    alpha: float,
    bundle: HierProbeBundle,
    device: torch.device,
) -> torch.Tensor:
    """Compute a fixed hidden-space direction from probe output weights."""
    dist = get_species_distribution_for_target(rank, target_group_idx, bundle).to(device)
    # probe.weight: (n_species, hidden_dim)
    direction = dist @ probe.weight.float()
    norm = direction.norm().clamp(min=1e-8)
    return (alpha * direction / norm).detach()


def _rank_nll_from_logits(
    logits: torch.Tensor,
    rank: str,
    target_group_idx: int,
    bundle: HierProbeBundle,
) -> torch.Tensor:
    n_pos = logits.shape[0]
    device = logits.device
    if rank == "species":
        labels = torch.full((n_pos,), int(target_group_idx), dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)

    rank_data = bundle.rank_data[rank]
    probs = torch.softmax(logits, dim=-1)
    group_probs = probs @ rank_data.M.to(device).T
    labels = torch.full((n_pos,), int(target_group_idx), dtype=torch.long, device=device)
    return F.nll_loss(torch.log(group_probs + 1e-8), labels)


def compute_adaptive_steering_vector(
    hidden: torch.Tensor,
    probe: nn.Linear,
    rank: str,
    target_group_idx: int,
    alpha: float,
    bundle: HierProbeBundle,
) -> torch.Tensor:
    """Compute gradient-based steering vector for provided hidden positions."""
    h = hidden.detach().float().clone().requires_grad_(True)
    logits = probe(h)
    loss = _rank_nll_from_logits(logits, rank, target_group_idx, bundle)
    grad = torch.autograd.grad(loss, h, retain_graph=False, create_graph=False)[0]
    return (-alpha * grad).detach()


def sample_context_from_og(
    seqs: list[str],
    taxids: list[int],
    n_max_protein: int,
    generator: torch.Generator,
) -> tuple[list[str], list[int], int, int]:
    """Subsample proteins like training and return context + last metadata."""
    n_keep = min(n_max_protein, len(seqs))
    idxs = torch.randperm(len(seqs), generator=generator)[:n_keep]
    sampled_seqs = [seqs[int(i)] for i in idxs]
    sampled_taxids = [int(taxids[int(i)]) for i in idxs]
    last_protein_len = len(sampled_seqs[-1])
    last_taxid = int(sampled_taxids[-1])
    return sampled_seqs, sampled_taxids, last_protein_len, last_taxid


def pick_wrong_group_idx(
    right_group_idx: int,
    rank: str,
    bundle: HierProbeBundle,
    rng: random.Random,
) -> int:
    if rank == "species":
        if bundle.n_species <= 1:
            raise RuntimeError("Cannot sample wrong species index with <=1 species")
        candidates = [i for i in range(bundle.n_species) if i != right_group_idx]
    else:
        n_groups = len(bundle.rank_data[rank].group_classes)
        if n_groups <= 1:
            raise RuntimeError(f"Cannot sample wrong group index for rank={rank} with <=1 group")
        candidates = [i for i in range(n_groups) if i != right_group_idx]
    return int(rng.choice(candidates))


class LastProteinSteeringHook:
    """Adds steering tensor to positions that predict last protein tokens."""

    def __init__(self, steering: torch.Tensor, last_protein_len: int):
        self.steering = steering
        self.last_protein_len = int(last_protein_len)

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        L = self.last_protein_len

        steer = self.steering
        if steer.ndim == 1:
            steer = steer.unsqueeze(0).expand(L, -1)
        steer = steer.to(hidden.device, dtype=hidden.dtype)

        hidden[:, -(L + 1) : -1, :] += steer
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


class GenerationLastTokenSteeringHook:
    """Adds a fixed steering direction to the last token at each decode step."""

    def __init__(self, steering: torch.Tensor):
        if steering.ndim != 1:
            raise ValueError("Generation steering vector must be 1D [hidden_dim]")
        self.steering = steering

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steer = self.steering.to(hidden.device, dtype=hidden.dtype)
        hidden[:, -1, :] += steer
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

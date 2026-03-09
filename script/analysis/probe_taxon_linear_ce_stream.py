"""
Train a per-position linear taxonomy probe with cross-entropy in PyTorch.

Supports four data modes:
  - stream:    load one training pickle file at a time (low RAM)
  - in_memory: preload all filtered hidden states + labels
  - mmap:      stream from sharded on-disk mmap cache
  - auto:      try in_memory, fall back to stream on memory failure

Only hidden representations and labels are retained for training cache.
Other pickle fields are dropped immediately.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import threading
import time
from collections import Counter, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Add at top of file
import mmap as _mmap_mod


def _drop_page_cache(path: Path) -> None:
    """Advise kernel to drop cached pages for this file."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


# Pickle schema keys
HIDDEN_KEY = "last_protein_hiddens"
TAXID_KEY = "lin"
LAYER_IDX = 0

DEFAULT_TRAIN_GLOB = (
    "/scratch/suyuelyu/deimm/data/oma/" "oma_probe_last_protein_hidden_lyr16_24_28*.pkl"
)
DEFAULT_VAL_FILES = "/scratch/suyuelyu/deimm/data/oma/oma_probe_last_protein_hidden_lyr16_24_28_32_val_0.pkl"
DEFAULT_TEST_FILES = "/scratch/suyuelyu/deimm/data/oma/oma_probe_last_protein_hidden_lyr16_24_28_32_test_0.pkl"
DEFAULT_TAXONOMY_MAPPING = "/scratch/suyuelyu/deimm/data/oma/taxid_to_std_ranks.pkl"
DEFAULT_OUTPUT_DIR = (
    "/scratch/suyuelyu/deimm/results/probe_taxon/linear_ce_stream_dualmode"
)
MMAP_CACHE_VERSION = 1


def save_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    """Save args as JSON and a snapshot of this script to output_dir."""
    # Save args
    args_path = output_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save a copy of this script
    script_path = Path(__file__).resolve()
    dst_path = output_dir / script_path.name
    import shutil

    shutil.copy2(script_path, dst_path)

    print(f"Saved run metadata: {args_path}, {dst_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear taxonomy probe with stream/in-memory/mmap data loading."
    )

    # Core
    parser.add_argument("--rank", type=str, required=True)
    parser.add_argument(
        "--taxonomy_mapping_file",
        type=str,
        default=DEFAULT_TAXONOMY_MAPPING,
    )
    parser.add_argument("--train_glob", type=str, default=DEFAULT_TRAIN_GLOB)
    parser.add_argument("--val_files", type=str, default=DEFAULT_VAL_FILES)
    parser.add_argument("--test_files", type=str, default=DEFAULT_TEST_FILES)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size_positions", type=int, default=2_500_000)
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default="balanced",
        choices=["balanced", "none", "log", "effective"],
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["none", "reduce_on_plateau"],
        help="Learning-rate scheduler type.",
    )
    parser.add_argument(
        "--lr_scheduler_patience",
        type=int,
        default=2,
        help="Scheduler steps to wait before reducing LR when train-step loss plateaus.",
    )
    parser.add_argument(
        "--lr_scheduler_factor",
        type=float,
        default=0.5,
        help="Multiplicative LR decay factor for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr",
        type=float,
        default=1e-6,
        help="Lower bound on LR for scheduler.",
    )
    parser.add_argument(
        "--lr_scheduler_threshold",
        type=float,
        default=1e-4,
        help="Minimum train-step-loss improvement to reset scheduler patience.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable mixed precision on CUDA"
    )
    parser.add_argument("--min_class_count", type=int, default=50)

    # Data mode
    parser.add_argument(
        "--data_mode",
        type=str,
        default="stream",
        choices=["stream", "in_memory", "mmap", "auto"],
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for mmap cache shards and cache_index.json.",
    )
    parser.add_argument(
        "--build_cache",
        action="store_true",
        help="Build (or rebuild) mmap cache before training.",
    )
    parser.add_argument(
        "--build_cache_only",
        action="store_true",
        help="Build mmap cache and exit without training.",
    )
    parser.add_argument(
        "--cache_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--max_train_files",
        type=int,
        default=None,
        help="Optional debug cap on number of train files after sorting.",
    )
    parser.add_argument(
        "--mmap_shard_rows",
        type=int,
        default=2_000_000,
        help="Rows per mmap shard when building cache.",
    )
    parser.add_argument(
        "--cache_num_workers",
        type=int,
        default=0,
        help=(
            "Parallel workers for class counting and mmap cache build. "
            "0 = auto(min(8, max(1, os.cpu_count()//2)))."
        ),
    )
    parser.add_argument(
        "--cache_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork"],
        help="Multiprocessing start method for parallel class counting/cache build.",
    )
    parser.add_argument(
        "--shuffle_mode",
        type=str,
        default="chunk",
        choices=["chunk", "global", "sequential"],
        help="Shuffle strategy for mmap mode.",
    )
    parser.add_argument(
        "--shuffle_block_rows",
        type=int,
        default=250_000,
        help="Rows per shuffle block for mmap chunk/global modes.",
    )
    parser.add_argument(
        "--shuffle_device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Where to run per-batch row shuffle (cpu or cuda).",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=1,
        help="Number of host batches to prefetch to device (0 disables prefetch).",
    )
    parser.add_argument(
        "--disk_prefetch_batches",
        type=int,
        default=0,
        help="Number of host batches to prefetch from disk iterator (0 disables).",
    )
    parser.add_argument(
        "--host_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Host dtype for mmap training batch buffer.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Run val evaluation every N epochs (always runs on final epoch).",
    )

    # Checkpointing
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint path (defaults to output_dir/checkpoint_last.pt if present).",
    )
    parser.add_argument("--save_every_epoch", action="store_true")

    return parser.parse_args()


def parse_csv_paths(arg: str) -> list[str]:
    paths = [p.strip() for p in arg.split(",") if p.strip()]
    return paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_hidden(prot: dict) -> np.ndarray:
    hidden = prot[HIDDEN_KEY]
    if LAYER_IDX is not None:
        hidden = hidden[LAYER_IDX]
    if hasattr(hidden, "numpy"):
        hidden = hidden.numpy()
    hidden = np.asarray(hidden)
    if hidden.ndim != 2:
        raise ValueError(f"Expected 2D hidden tensor, got shape {hidden.shape}")
    return hidden


def load_rank_mapping(mapping_file: str, rank: str) -> dict[int, int]:
    with open(mapping_file, "rb") as f:
        taxid_to_std_ranks = pickle.load(f)

    rank_mapping: dict[int, int] = {}
    for species_tid, rank_dict in taxid_to_std_ranks.items():
        if rank in rank_dict:
            rank_mapping[int(species_tid)] = int(rank_dict[rank])

    if not rank_mapping:
        raise ValueError(f"No entries found for rank '{rank}' in {mapping_file}")

    n_labels = len(set(rank_mapping.values()))
    print(
        f"Loaded rank mapping for '{rank}': "
        f"{len(rank_mapping):,} species mapped, {n_labels:,} unique labels"
    )
    return rank_mapping


def collect_train_files(train_glob: str, max_train_files: int | None) -> list[str]:
    files = sorted(glob.glob(train_glob))
    if not files:
        raise FileNotFoundError(f"No train files match glob: {train_glob}")

    if max_train_files is not None:
        files = files[:max_train_files]

    print(f"Using {len(files):,} training files")
    return files


def extract_sample(
    prot: dict,
    tid_to_class_idx: dict[int, int],
) -> tuple[np.ndarray, int] | None:
    tid = int(prot[TAXID_KEY])
    cls_idx = tid_to_class_idx.get(tid)
    if cls_idx is None:
        return None
    hidden = get_hidden(prot)
    return hidden, cls_idx


def count_classes_and_hidden_dim(
    train_files: list[str],
    rank_mapping: dict[int, int],
    count_num_workers: int = 1,
    count_start_method: str = "spawn",
) -> tuple[Counter, Counter, int]:
    resolved_workers = resolve_cache_num_workers(count_num_workers, len(train_files))
    if resolved_workers <= 1:
        worker_result = _count_classes_worker(
            worker_id=0,
            assigned_files=list(train_files),
            rank_mapping=rank_mapping,
        )
        hidden_dim = (
            int(worker_result["hidden_dim"])
            if worker_result["hidden_dim"] is not None
            else None
        )
        if hidden_dim is None:
            raise RuntimeError("Could not infer hidden dimension from training data.")
        return (
            Counter(worker_result["protein_counts"]),
            Counter(worker_result["position_counts"]),
            hidden_dim,
        )

    sorted_files = sorted(train_files)
    assignments: list[list[str]] = [[] for _ in range(resolved_workers)]
    for file_idx, pf in enumerate(sorted_files):
        assignments[file_idx % resolved_workers].append(pf)
    active_assignments = [
        (worker_id, files)
        for worker_id, files in enumerate(assignments)
        if len(files) > 0
    ]
    if len(active_assignments) <= 1:
        worker_result = _count_classes_worker(
            worker_id=0,
            assigned_files=list(train_files),
            rank_mapping=rank_mapping,
        )
        hidden_dim = (
            int(worker_result["hidden_dim"])
            if worker_result["hidden_dim"] is not None
            else None
        )
        if hidden_dim is None:
            raise RuntimeError("Could not infer hidden dimension from training data.")
        return (
            Counter(worker_result["protein_counts"]),
            Counter(worker_result["position_counts"]),
            hidden_dim,
        )

    print(
        f"Counting classes in parallel with {len(active_assignments)} workers "
        f"(start_method={count_start_method})"
    )
    try:
        ctx = mp.get_context(count_start_method)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported multiprocessing start method '{count_start_method}' on this platform."
        ) from exc

    worker_results: dict[int, dict[str, Any]] = {}
    with cf.ProcessPoolExecutor(
        max_workers=len(active_assignments), mp_context=ctx
    ) as executor:
        future_to_worker: dict[cf.Future, int] = {}
        for worker_id, files in active_assignments:
            future = executor.submit(
                _count_classes_worker,
                worker_id,
                files,
                rank_mapping,
            )
            future_to_worker[future] = worker_id

        for future in tqdm(
            cf.as_completed(future_to_worker),
            total=len(future_to_worker),
            desc="Counting classes (workers)",
        ):
            worker_id = future_to_worker[future]
            try:
                worker_results[worker_id] = future.result()
            except Exception as exc:
                print(
                    f"Class-count worker {worker_id} failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise

    protein_counts: Counter = Counter()
    position_counts: Counter = Counter()
    hidden_dim: int | None = None
    for worker_id, _ in active_assignments:
        wr = worker_results[worker_id]
        protein_counts.update(Counter(wr["protein_counts"]))
        position_counts.update(Counter(wr["position_counts"]))
        worker_hidden = wr.get("hidden_dim")
        if worker_hidden is None:
            continue
        worker_hidden_i = int(worker_hidden)
        if hidden_dim is None:
            hidden_dim = worker_hidden_i
        elif worker_hidden_i != hidden_dim:
            raise ValueError(
                f"Inconsistent hidden dim across workers: "
                f"worker {worker_id} has {worker_hidden_i} vs expected {hidden_dim}"
            )

    if hidden_dim is None:
        raise RuntimeError("Could not infer hidden dimension from training data.")

    return protein_counts, position_counts, hidden_dim


def _count_classes_worker(
    worker_id: int,
    assigned_files: list[str],
    rank_mapping: dict[int, int],
) -> dict[str, Any]:
    protein_counts: Counter = Counter()
    position_counts: Counter = Counter()
    hidden_dim: int | None = None
    files_processed = 0

    for pf in (
        tqdm(assigned_files, desc=f"Worker {worker_id} counting files")
        if worker_id == 0
        else assigned_files
    ):
        path = Path(pf)
        if not path.exists():
            print(f"  WARNING(count-worker={worker_id}): missing train file: {path}")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)
        files_processed += 1

        for prot in data:
            tid = int(prot[TAXID_KEY])
            label = rank_mapping.get(tid)
            if label is None:
                continue

            hidden = get_hidden(prot)
            if hidden_dim is None:
                hidden_dim = int(hidden.shape[1])
            elif hidden.shape[1] != hidden_dim:
                raise ValueError(
                    f"Inconsistent hidden dim in {path}: "
                    f"got {hidden.shape[1]} vs expected {hidden_dim}"
                )

            protein_counts[int(label)] += 1
            position_counts[int(label)] += int(hidden.shape[0])

        del data
        gc.collect()

    return {
        "worker_id": int(worker_id),
        "files_processed": int(files_processed),
        "protein_counts": dict(protein_counts),
        "position_counts": dict(position_counts),
        "hidden_dim": hidden_dim,
    }


def hash_train_files(train_files: list[str]) -> str:
    h = hashlib.sha256()
    for pf in sorted(train_files):
        path = Path(pf)
        h.update(str(path).encode("utf-8"))
        if path.exists():
            st = path.stat()
            h.update(f":{st.st_size}:{st.st_mtime_ns};".encode("ascii"))
        else:
            h.update(b":missing;")
    return h.hexdigest()


def hash_rank_mapping(rank_mapping: dict[int, int]) -> str:
    h = hashlib.sha256()
    for tid, label in sorted(rank_mapping.items()):
        h.update(f"{int(tid)}:{int(label)};".encode("ascii"))
    return h.hexdigest()


def build_or_load_counts(
    output_dir: Path,
    rank: str,
    min_class_count: int,
    train_files: list[str],
    rank_mapping: dict[int, int],
    count_num_workers: int,
    count_start_method: str,
) -> tuple[set[int], Counter, Counter, int]:
    counts_path = output_dir / f"counts_{rank}_min{min_class_count}.pkl"
    expected_train_files_hash = hash_train_files(train_files)
    expected_rank_mapping_hash = hash_rank_mapping(rank_mapping)

    use_cached_counts = False
    payload: dict[str, Any] | None = None
    if counts_path.exists():
        with open(counts_path, "rb") as f:
            payload = pickle.load(f)
        cache_train_hash = str(payload.get("train_files_hash", ""))
        cache_rank_hash = str(payload.get("rank_mapping_hash", ""))
        cache_rank = str(payload.get("rank", ""))
        cache_min_class_count = int(payload.get("min_class_count", -1))
        if (
            cache_train_hash == expected_train_files_hash
            and cache_rank_hash == expected_rank_mapping_hash
            and cache_rank == rank
            and cache_min_class_count == int(min_class_count)
            and "valid_classes" in payload
            and "protein_counts" in payload
            and "position_counts" in payload
            and "hidden_dim" in payload
        ):
            use_cached_counts = True
        else:
            print(
                f"Cached class counts at {counts_path} are stale or incompatible; "
                "recomputing counts."
            )

    if use_cached_counts:
        assert payload is not None
        valid_classes = set(payload["valid_classes"])
        protein_counts = Counter(payload["protein_counts"])
        position_counts = Counter(payload["position_counts"])
        hidden_dim = int(payload["hidden_dim"])
        print(f"Loaded cached class counts from {counts_path}")
    else:
        protein_counts, position_counts, hidden_dim = count_classes_and_hidden_dim(
            train_files=train_files,
            rank_mapping=rank_mapping,
            count_num_workers=count_num_workers,
            count_start_method=count_start_method,
        )
        valid_classes = {c for c, n in protein_counts.items() if n >= min_class_count}
        payload = {
            "valid_classes": sorted(valid_classes),
            "protein_counts": dict(protein_counts),
            "position_counts": dict(position_counts),
            "hidden_dim": hidden_dim,
            "rank": rank,
            "min_class_count": int(min_class_count),
            "train_files_hash": expected_train_files_hash,
            "rank_mapping_hash": expected_rank_mapping_hash,
        }
        with open(counts_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved class counts to {counts_path}")

    if len(valid_classes) < 2:
        raise RuntimeError(
            f"Need at least 2 classes after filtering (got {len(valid_classes)}). "
            "Lower --min_class_count or choose another rank."
        )

    total_proteins = sum(protein_counts[c] for c in valid_classes)
    total_positions = sum(position_counts[c] for c in valid_classes)
    print(
        f"Rank summary: {len(valid_classes):,} classes, "
        f"{total_proteins:,} proteins, {total_positions:,} positions"
    )
    return valid_classes, protein_counts, position_counts, hidden_dim


def resolve_cache_dir(
    cache_dir_arg: str | None,
    output_dir: Path,
    rank: str,
    min_class_count: int,
    cache_dtype: str,
) -> Path:
    if cache_dir_arg is not None and cache_dir_arg.strip():
        return Path(cache_dir_arg)
    return output_dir / f"mmap_cache_{rank}_min{min_class_count}_{cache_dtype}"


def resolve_cache_num_workers(cache_num_workers: int, n_train_files: int) -> int:
    if cache_num_workers < 0:
        raise ValueError("--cache_num_workers must be >= 0")
    if n_train_files <= 0:
        return 1
    workers = int(cache_num_workers)
    if workers == 0:
        cpu_count = os.cpu_count() or 1
        workers = min(8, max(1, cpu_count // 2))
    return max(1, min(workers, n_train_files))


def cache_dtype_suffix(cache_dtype: np.dtype) -> str:
    dtype = np.dtype(cache_dtype)
    if dtype == np.float16:
        return "f16"
    if dtype == np.float32:
        return "f32"
    raise ValueError(f"Unsupported cache dtype: {dtype}")


def class_mapping_hash(classes: np.ndarray, tid_to_class_idx: dict[int, int]) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(classes, dtype=np.int64).tobytes())
    for tid, cls_idx in sorted(tid_to_class_idx.items()):
        h.update(f"{tid}:{cls_idx};".encode("ascii"))
    return h.hexdigest()


def load_cache_index(cache_dir: Path) -> dict[str, Any]:
    index_path = cache_dir / "cache_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing cache index: {index_path}")
    with open(index_path, "r") as f:
        return dict(json.load(f))


def cache_dtype_from_index(index: dict[str, Any]) -> np.dtype:
    dtype_name = str(index.get("cache_dtype", "")).strip()
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported cache_dtype in index: {dtype_name!r}")


def cleanup_cache_shards_from_index(cache_dir: Path, index: dict[str, Any]) -> int:
    removed = 0
    for shard in list(index.get("shards", [])):
        for key in ("x_file", "y_file"):
            file_name = str(shard.get(key, "")).strip()
            if not file_name:
                continue
            path = cache_dir / file_name
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def validate_cache_index(
    cache_dir: Path,
    index: dict[str, Any],
    hidden_dim: int,
    n_cls: int,
    expected_class_hist: np.ndarray,
    expected_class_hash: str,
    expected_classes: np.ndarray,
) -> None:
    version = int(index.get("version", -1))
    if version != MMAP_CACHE_VERSION:
        raise ValueError(
            f"Unsupported cache version {version}; expected {MMAP_CACHE_VERSION}"
        )

    if int(index.get("hidden_dim", -1)) != int(hidden_dim):
        raise ValueError(
            f"Cache hidden_dim mismatch: {index.get('hidden_dim')} vs expected {hidden_dim}"
        )
    if int(index.get("n_classes", -1)) != int(n_cls):
        raise ValueError(
            f"Cache n_classes mismatch: {index.get('n_classes')} vs expected {n_cls}"
        )

    cache_hash = str(index.get("class_mapping_hash", ""))
    if cache_hash != expected_class_hash:
        raise ValueError(
            "Cache class_mapping_hash mismatch. Cache was built with a different class mapping."
        )

    classes_list = [int(x) for x in index.get("classes", [])]
    expected_classes_list = [int(x) for x in np.asarray(expected_classes).tolist()]
    if classes_list != expected_classes_list:
        raise ValueError("Cache classes list mismatch.")

    class_hist = np.asarray(index.get("class_histogram_rows", []), dtype=np.int64)
    if class_hist.shape != expected_class_hist.shape:
        raise ValueError(
            f"Cache class histogram shape mismatch: {class_hist.shape} "
            f"vs expected {expected_class_hist.shape}"
        )
    if not np.array_equal(class_hist, expected_class_hist):
        raise ValueError(
            "Cache class histogram does not match expected position counts."
        )

    shards = list(index.get("shards", []))
    if len(shards) == 0:
        raise ValueError("Cache index has zero shards.")

    cache_dtype = cache_dtype_from_index(index)
    dtype_bytes = int(cache_dtype.itemsize)
    label_bytes = int(np.dtype(np.int32).itemsize)
    rows_sum = 0

    for shard in shards:
        rows = int(shard["rows"])
        x_file = cache_dir / str(shard["x_file"])
        y_file = cache_dir / str(shard["y_file"])
        if rows <= 0:
            raise ValueError(f"Invalid shard rows={rows} in {shard}")
        if not x_file.exists() or not y_file.exists():
            raise FileNotFoundError(f"Missing shard files for {shard}")

        expected_x_size = rows * int(hidden_dim) * dtype_bytes
        expected_y_size = rows * label_bytes
        actual_x_size = int(os.path.getsize(x_file))
        actual_y_size = int(os.path.getsize(y_file))
        if actual_x_size != expected_x_size:
            raise ValueError(
                f"Shard X size mismatch for {x_file}: {actual_x_size} vs {expected_x_size}"
            )
        if actual_y_size != expected_y_size:
            raise ValueError(
                f"Shard y size mismatch for {y_file}: {actual_y_size} vs {expected_y_size}"
            )
        rows_sum += rows

    total_rows = int(index.get("total_rows", -1))
    if total_rows != rows_sum:
        raise ValueError(
            f"Cache total_rows mismatch: {total_rows} vs computed {rows_sum}"
        )
    expected_total_rows = int(expected_class_hist.sum())
    if rows_sum != expected_total_rows:
        raise ValueError(
            f"Cache row count mismatch: {rows_sum} vs expected {expected_total_rows}"
        )


def _build_mmap_cache_worker(
    worker_id: int,
    assigned_files: list[str],
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    n_cls: int,
    cache_dir_str: str,
    cache_dtype_name: str,
    mmap_shard_rows: int,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir_str)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dtype = np.dtype(cache_dtype_name)
    shard_suffix = cache_dtype_suffix(shard_dtype)
    class_hist = np.zeros((n_cls,), dtype=np.int64)

    shards: list[dict[str, Any]] = []
    total_rows = 0
    files_processed = 0
    local_shard_idx = 0

    x_mm: np.memmap | None = None
    y_mm: np.memmap | None = None
    x_path: Path | None = None
    y_path: Path | None = None
    fill = 0

    def open_next_shard() -> None:
        nonlocal x_mm, y_mm, x_path, y_path, fill, local_shard_idx
        x_path = cache_dir / (
            f"train_X_w{worker_id:03d}_s{local_shard_idx:05d}.{shard_suffix}.bin"
        )
        y_path = cache_dir / f"train_y_w{worker_id:03d}_s{local_shard_idx:05d}.i32.bin"
        x_mm = np.memmap(
            x_path,
            mode="w+",
            dtype=shard_dtype,
            shape=(mmap_shard_rows, hidden_dim),
        )
        y_mm = np.memmap(
            y_path,
            mode="w+",
            dtype=np.int32,
            shape=(mmap_shard_rows,),
        )
        fill = 0

    def close_current_shard(rows_written: int) -> None:
        nonlocal x_mm, y_mm, x_path, y_path, local_shard_idx
        if x_mm is None or y_mm is None or x_path is None or y_path is None:
            return

        x_mm.flush()
        y_mm.flush()
        del x_mm
        del y_mm
        x_mm = None
        y_mm = None

        if rows_written < mmap_shard_rows:
            with open(x_path, "r+b") as f:
                f.truncate(rows_written * hidden_dim * shard_dtype.itemsize)
            with open(y_path, "r+b") as f:
                f.truncate(rows_written * np.dtype(np.int32).itemsize)

        shards.append(
            {
                "rows": int(rows_written),
                "x_file": x_path.name,
                "y_file": y_path.name,
                "worker_id": int(worker_id),
                "local_shard_idx": int(local_shard_idx),
            }
        )
        local_shard_idx += 1

    t0 = time.time()
    for pf in (
        tqdm(assigned_files, desc=f"Worker {worker_id} processing files")
        if worker_id == 0
        else assigned_files
    ):
        path = Path(pf)
        if not path.exists():
            print(f"  WARNING(worker={worker_id}): missing train file: {path}")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)
        files_processed += 1

        for prot in data:
            sample = extract_sample(prot, tid_to_class_idx)
            if sample is None:
                continue
            hidden, cls_idx = sample
            if hidden.shape[1] != hidden_dim:
                raise ValueError(
                    f"Inconsistent hidden dim in {path}: "
                    f"{hidden.shape[1]} vs expected {hidden_dim}"
                )

            hidden_np = np.asarray(hidden, dtype=shard_dtype, order="C")
            start = 0
            total = int(hidden_np.shape[0])
            while start < total:
                if x_mm is None or y_mm is None:
                    open_next_shard()
                room = mmap_shard_rows - fill
                take = min(room, total - start)
                x_mm[fill : fill + take] = hidden_np[start : start + take]
                y_mm[fill : fill + take] = int(cls_idx)
                fill += take
                start += take
                total_rows += take
                class_hist[int(cls_idx)] += take

                if fill == mmap_shard_rows:
                    close_current_shard(rows_written=fill)
                    fill = 0

        del data
        gc.collect()

    if x_mm is not None and y_mm is not None and fill > 0:
        close_current_shard(rows_written=fill)

    return {
        "worker_id": int(worker_id),
        "total_rows": int(total_rows),
        "class_histogram_rows": [int(x) for x in class_hist.tolist()],
        "files_processed": int(files_processed),
        "elapsed_sec": float(time.time() - t0),
        "shards": shards,
    }


def build_mmap_cache_serial(
    train_files: list[str],
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    n_cls: int,
    classes: np.ndarray,
    expected_class_hist: np.ndarray,
    expected_class_hash: str,
    cache_dir: Path,
    cache_dtype: np.dtype,
    mmap_shard_rows: int,
    rank: str,
    min_class_count: int,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker_result = _build_mmap_cache_worker(
        worker_id=0,
        assigned_files=list(train_files),
        tid_to_class_idx=tid_to_class_idx,
        hidden_dim=hidden_dim,
        n_cls=n_cls,
        cache_dir_str=str(cache_dir),
        cache_dtype_name=np.dtype(cache_dtype).name,
        mmap_shard_rows=mmap_shard_rows,
    )

    class_hist = np.asarray(worker_result["class_histogram_rows"], dtype=np.int64)
    total_rows = int(worker_result["total_rows"])
    if total_rows <= 0:
        raise RuntimeError("Built mmap cache has zero rows.")

    expected_total = int(expected_class_hist.sum())
    if int(class_hist.sum()) != expected_total:
        raise RuntimeError(
            "Cache build row mismatch: "
            f"{int(class_hist.sum())} vs expected {expected_total}. "
            "This often indicates stale cached counts; rerun so counts are recomputed."
        )
    if not np.array_equal(class_hist, expected_class_hist):
        raise RuntimeError(
            "Cache build class histogram mismatch against expected position counts."
        )

    shards = sorted(
        list(worker_result["shards"]),
        key=lambda s: (int(s["worker_id"]), int(s["local_shard_idx"])),
    )
    for shard_idx, shard in enumerate(shards):
        shard["shard_idx"] = int(shard_idx)

    index = {
        "version": MMAP_CACHE_VERSION,
        "rank": rank,
        "min_class_count": int(min_class_count),
        "hidden_dim": int(hidden_dim),
        "n_classes": int(n_cls),
        "cache_dtype": str(np.dtype(cache_dtype).name),
        "label_dtype": "int32",
        "mmap_shard_rows": int(mmap_shard_rows),
        "total_rows": int(total_rows),
        "class_histogram_rows": [int(x) for x in class_hist.tolist()],
        "classes": [int(x) for x in np.asarray(classes).tolist()],
        "class_mapping_hash": expected_class_hash,
        "build_time_sec": float(worker_result["elapsed_sec"]),
        "build_num_workers": 1,
        "build_strategy": "serial_single_process",
        "worker_summaries": [
            {
                "worker_id": int(worker_result["worker_id"]),
                "files_processed": int(worker_result["files_processed"]),
                "total_rows": int(worker_result["total_rows"]),
                "elapsed_sec": float(worker_result["elapsed_sec"]),
                "num_shards": int(len(shards)),
            }
        ],
        "shards": shards,
    }
    index_path = cache_dir / "cache_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(
        f"Built mmap cache (serial) at {cache_dir} with {len(shards):,} shards, "
        f"{total_rows:,} rows, {index['build_time_sec']:.1f}s"
    )
    return index


def build_mmap_cache_parallel(
    train_files: list[str],
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    n_cls: int,
    classes: np.ndarray,
    expected_class_hist: np.ndarray,
    expected_class_hash: str,
    cache_dir: Path,
    cache_dtype: np.dtype,
    mmap_shard_rows: int,
    rank: str,
    min_class_count: int,
    cache_num_workers: int,
    cache_start_method: str,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sorted_files = sorted(train_files)
    assignments: list[list[str]] = [[] for _ in range(cache_num_workers)]
    for file_idx, pf in enumerate(sorted_files):
        assignments[file_idx % cache_num_workers].append(pf)
    active_assignments = [
        (worker_id, files)
        for worker_id, files in enumerate(assignments)
        if len(files) > 0
    ]
    if len(active_assignments) <= 1:
        return build_mmap_cache_serial(
            train_files=train_files,
            tid_to_class_idx=tid_to_class_idx,
            hidden_dim=hidden_dim,
            n_cls=n_cls,
            classes=classes,
            expected_class_hist=expected_class_hist,
            expected_class_hash=expected_class_hash,
            cache_dir=cache_dir,
            cache_dtype=cache_dtype,
            mmap_shard_rows=mmap_shard_rows,
            rank=rank,
            min_class_count=min_class_count,
        )

    print(
        f"Building mmap cache in parallel with {len(active_assignments)} workers "
        f"(start_method={cache_start_method})"
    )
    t0 = time.time()
    try:
        ctx = mp.get_context(cache_start_method)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported multiprocessing start method '{cache_start_method}' on this platform."
        ) from exc
    worker_results: dict[int, dict[str, Any]] = {}

    with cf.ProcessPoolExecutor(
        max_workers=len(active_assignments), mp_context=ctx
    ) as executor:
        future_to_worker: dict[cf.Future, int] = {}
        for worker_id, files in active_assignments:
            future = executor.submit(
                _build_mmap_cache_worker,
                worker_id,
                files,
                tid_to_class_idx,
                hidden_dim,
                n_cls,
                str(cache_dir),
                np.dtype(cache_dtype).name,
                mmap_shard_rows,
            )
            future_to_worker[future] = worker_id

        for future in tqdm(
            cf.as_completed(future_to_worker),
            total=len(future_to_worker),
            desc="Building mmap cache (workers)",
        ):
            worker_id = future_to_worker[future]
            try:
                worker_results[worker_id] = future.result()
            except Exception as exc:
                print(
                    f"Cache build worker {worker_id} failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                raise

    ordered_results = [worker_results[wid] for wid, _ in sorted(active_assignments)]
    class_hist = np.zeros((n_cls,), dtype=np.int64)
    total_rows = 0
    shards: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for worker_result in ordered_results:
        worker_hist = np.asarray(worker_result["class_histogram_rows"], dtype=np.int64)
        if worker_hist.shape != class_hist.shape:
            raise RuntimeError(
                "Worker histogram shape mismatch during parallel cache aggregation."
            )
        class_hist += worker_hist
        total_rows += int(worker_result["total_rows"])
        worker_shards = list(worker_result["shards"])
        shards.extend(worker_shards)
        worker_summaries.append(
            {
                "worker_id": int(worker_result["worker_id"]),
                "files_processed": int(worker_result["files_processed"]),
                "total_rows": int(worker_result["total_rows"]),
                "elapsed_sec": float(worker_result["elapsed_sec"]),
                "num_shards": int(len(worker_shards)),
            }
        )

    if total_rows <= 0:
        raise RuntimeError("Built mmap cache has zero rows.")
    expected_total = int(expected_class_hist.sum())
    if int(class_hist.sum()) != expected_total:
        raise RuntimeError(
            "Cache build row mismatch: "
            f"{int(class_hist.sum())} vs expected {expected_total}. "
            "This often indicates stale cached counts; rerun so counts are recomputed."
        )
    if not np.array_equal(class_hist, expected_class_hist):
        raise RuntimeError(
            "Cache build class histogram mismatch against expected position counts."
        )

    shards.sort(key=lambda s: (int(s["worker_id"]), int(s["local_shard_idx"])))
    for shard_idx, shard in enumerate(shards):
        shard["shard_idx"] = int(shard_idx)

    index = {
        "version": MMAP_CACHE_VERSION,
        "rank": rank,
        "min_class_count": int(min_class_count),
        "hidden_dim": int(hidden_dim),
        "n_classes": int(n_cls),
        "cache_dtype": str(np.dtype(cache_dtype).name),
        "label_dtype": "int32",
        "mmap_shard_rows": int(mmap_shard_rows),
        "total_rows": int(total_rows),
        "class_histogram_rows": [int(x) for x in class_hist.tolist()],
        "classes": [int(x) for x in np.asarray(classes).tolist()],
        "class_mapping_hash": expected_class_hash,
        "build_time_sec": float(time.time() - t0),
        "build_num_workers": int(len(active_assignments)),
        "build_strategy": "parallel_per_worker_shards",
        "worker_summaries": worker_summaries,
        "shards": shards,
    }
    index_path = cache_dir / "cache_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(
        f"Built mmap cache (parallel) at {cache_dir} with {len(shards):,} shards, "
        f"{total_rows:,} rows, {index['build_time_sec']:.1f}s"
    )
    return index


def build_or_load_mmap_cache(
    train_files: list[str],
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    n_cls: int,
    classes: np.ndarray,
    expected_class_hist: np.ndarray,
    expected_class_hash: str,
    cache_dir: Path,
    cache_dtype: np.dtype,
    mmap_shard_rows: int,
    cache_num_workers: int,
    cache_start_method: str,
    rank: str,
    min_class_count: int,
    force_rebuild: bool,
    build_if_missing: bool,
) -> dict[str, Any]:
    index_path = cache_dir / "cache_index.json"
    if index_path.exists() and not force_rebuild:
        index = load_cache_index(cache_dir)
        index_dtype = cache_dtype_from_index(index)
        if index_dtype != np.dtype(cache_dtype):
            raise ValueError(
                f"Cache dtype mismatch in {index_path}: {index_dtype.name} "
                f"vs requested {np.dtype(cache_dtype).name}. "
                "Rebuild with --build_cache or change --cache_dtype."
            )
        validate_cache_index(
            cache_dir=cache_dir,
            index=index,
            hidden_dim=hidden_dim,
            n_cls=n_cls,
            expected_class_hist=expected_class_hist,
            expected_class_hash=expected_class_hash,
            expected_classes=classes,
        )
        print(f"Loaded valid mmap cache index from {index_path}")
        return index

    if not index_path.exists() and not build_if_missing and not force_rebuild:
        raise FileNotFoundError(
            f"mmap cache does not exist at {index_path}. "
            "Use --build_cache or switch data mode."
        )

    if force_rebuild and index_path.exists():
        removed = 0
        try:
            stale_index = load_cache_index(cache_dir)
            removed = cleanup_cache_shards_from_index(cache_dir, stale_index)
        except Exception as exc:
            print(
                f"WARNING: failed to clean stale cache shards from {index_path}: "
                f"{exc.__class__.__name__}: {exc}"
            )
        index_path.unlink(missing_ok=True)
        print(
            f"Rebuild requested: removed {removed} stale shard files from {cache_dir}"
        )

    resolved_workers = resolve_cache_num_workers(cache_num_workers, len(train_files))
    if resolved_workers <= 1:
        index = build_mmap_cache_serial(
            train_files=train_files,
            tid_to_class_idx=tid_to_class_idx,
            hidden_dim=hidden_dim,
            n_cls=n_cls,
            classes=classes,
            expected_class_hist=expected_class_hist,
            expected_class_hash=expected_class_hash,
            cache_dir=cache_dir,
            cache_dtype=cache_dtype,
            mmap_shard_rows=mmap_shard_rows,
            rank=rank,
            min_class_count=min_class_count,
        )
    else:
        index = build_mmap_cache_parallel(
            train_files=train_files,
            tid_to_class_idx=tid_to_class_idx,
            hidden_dim=hidden_dim,
            n_cls=n_cls,
            classes=classes,
            expected_class_hist=expected_class_hist,
            expected_class_hash=expected_class_hash,
            cache_dir=cache_dir,
            cache_dtype=cache_dtype,
            mmap_shard_rows=mmap_shard_rows,
            rank=rank,
            min_class_count=min_class_count,
            cache_num_workers=resolved_workers,
            cache_start_method=cache_start_method,
        )
    validate_cache_index(
        cache_dir=cache_dir,
        index=index,
        hidden_dim=hidden_dim,
        n_cls=n_cls,
        expected_class_hist=expected_class_hist,
        expected_class_hash=expected_class_hash,
        expected_classes=classes,
    )
    return index


def make_host_batch_from_buffer(
    x_buf: np.ndarray,
    y_buf: np.ndarray,
    n_rows: int,
    rng: np.random.Generator,
    shuffle_on_cpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0 for batch creation")
    if n_rows > 1 and shuffle_on_cpu:
        perm = rng.permutation(n_rows)
        x_np = x_buf[:n_rows][perm].copy()
        y_np = y_buf[:n_rows][perm].copy()
    else:
        x_np = x_buf[:n_rows].copy()
        y_np = y_buf[:n_rows].copy()
    return x_np, y_np


def iter_host_batches_prefetch(
    host_batch_iter: Iterator[tuple[np.ndarray, np.ndarray]],
    disk_prefetch_batches: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    prefetch_n = max(0, int(disk_prefetch_batches))
    if prefetch_n == 0:
        yield from host_batch_iter
        return

    host_iter = iter(host_batch_iter)
    q: queue.Queue[object] = queue.Queue(maxsize=prefetch_n)
    end_marker = object()

    def producer() -> None:
        try:
            for batch in host_iter:
                q.put(batch)
        except BaseException as exc:  # pragma: no cover - propagates to consumer
            q.put(exc)
            return
        q.put(end_marker)

    producer_thread = threading.Thread(
        target=producer,
        name="disk-host-prefetch",
        daemon=True,
    )
    producer_thread.start()
    try:
        while True:
            item = q.get()
            if item is end_marker:
                break
            if isinstance(item, BaseException):
                raise item
            x_np, y_np = item
            yield x_np, y_np
    finally:
        producer_thread.join(timeout=1.0)


def host_batch_to_device(
    x_np: np.ndarray,
    y_np: np.ndarray,
    device: torch.device,
    x_device_dtype: torch.dtype,
    pin_memory: bool,
    shuffle_on_cuda: bool,
    cuda_shuffle_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_host = torch.from_numpy(x_np)
    y_host = torch.from_numpy(y_np)
    if pin_memory:
        x_host = x_host.pin_memory()
        y_host = y_host.pin_memory()

    x_t = x_host.to(device=device, dtype=x_device_dtype, non_blocking=True)
    y_t = y_host.to(device=device, dtype=torch.long, non_blocking=True)

    if shuffle_on_cuda and x_t.shape[0] > 1:
        perm = torch.randperm(
            x_t.shape[0], device=device, generator=cuda_shuffle_generator
        )
        x_t = x_t.index_select(0, perm)
        y_t = y_t.index_select(0, perm)

    return x_t, y_t


def iter_device_batches_prefetch(
    host_batch_iter: Iterator[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    x_device_dtype: torch.dtype,
    shuffle_on_cuda: bool,
    cuda_shuffle_generator: torch.Generator | None,
    prefetch_batches: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    host_iter = iter(host_batch_iter)
    pin_memory = device.type == "cuda"
    prefetch_n = max(0, int(prefetch_batches))

    if device.type != "cuda" or prefetch_n == 0:
        for x_np, y_np in host_iter:
            yield host_batch_to_device(
                x_np=x_np,
                y_np=y_np,
                device=device,
                x_device_dtype=x_device_dtype,
                pin_memory=pin_memory,
                shuffle_on_cuda=shuffle_on_cuda,
                cuda_shuffle_generator=cuda_shuffle_generator,
            )
        return

    transfer_stream = torch.cuda.Stream(device=device)
    queue: deque[tuple[torch.Tensor, torch.Tensor]] = deque()

    def enqueue(x_np: np.ndarray, y_np: np.ndarray) -> None:
        with torch.cuda.stream(transfer_stream):
            x_t, y_t = host_batch_to_device(
                x_np=x_np,
                y_np=y_np,
                device=device,
                x_device_dtype=x_device_dtype,
                pin_memory=pin_memory,
                shuffle_on_cuda=shuffle_on_cuda,
                cuda_shuffle_generator=cuda_shuffle_generator,
            )
        queue.append((x_t, y_t))

    for _ in range(prefetch_n):
        try:
            x_np, y_np = next(host_iter)
        except StopIteration:
            break
        enqueue(x_np, y_np)

    while queue:
        torch.cuda.current_stream(device=device).wait_stream(transfer_stream)
        x_t, y_t = queue.popleft()
        x_t.record_stream(torch.cuda.current_stream(device=device))
        y_t.record_stream(torch.cuda.current_stream(device=device))

        try:
            x_np, y_np = next(host_iter)
            enqueue(x_np, y_np)
        except StopIteration:
            pass

        yield x_t, y_t


def iter_train_batches_stream(
    train_files: list[str],
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    batch_size_positions: int,
    rng: np.random.Generator,
    epoch_idx: int,
    shuffle_on_cpu: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    file_order = rng.permutation(len(train_files))
    file_preview = [Path(train_files[i]).name for i in file_order[:5]]
    print(f"Epoch {epoch_idx + 1} file shuffle preview: {file_preview}")

    x_buf = np.empty((batch_size_positions, hidden_dim), dtype=np.float32)
    y_buf = np.empty((batch_size_positions,), dtype=np.int64)
    fill = 0
    logged_protein_preview = False

    for file_idx in tqdm(file_order, desc=f"Epoch {epoch_idx + 1} stream files"):
        pf = Path(train_files[int(file_idx)])
        if not pf.exists():
            print(f"  WARNING: missing train file: {pf}")
            continue

        with open(pf, "rb") as f:
            data = pickle.load(f)

        protein_order = rng.permutation(len(data))
        if not logged_protein_preview and len(protein_order) > 0:
            print(
                f"Epoch {epoch_idx + 1} protein index preview ({pf.name}): "
                f"{protein_order[:8].tolist()}"
            )
            logged_protein_preview = True

        for protein_idx in protein_order:
            sample = extract_sample(data[int(protein_idx)], tid_to_class_idx)
            if sample is None:
                continue
            hidden, cls_idx = sample

            if hidden.shape[1] != hidden_dim:
                raise ValueError(
                    f"Inconsistent hidden dim in {pf}: "
                    f"{hidden.shape[1]} vs expected {hidden_dim}"
                )

            hidden = np.asarray(hidden, dtype=np.float32)
            start = 0
            total = hidden.shape[0]
            while start < total:
                room = batch_size_positions - fill
                take = min(room, total - start)
                x_buf[fill : fill + take] = hidden[start : start + take]
                y_buf[fill : fill + take] = cls_idx
                fill += take
                start += take

                if fill == batch_size_positions:
                    yield make_host_batch_from_buffer(
                        x_buf=x_buf,
                        y_buf=y_buf,
                        n_rows=fill,
                        rng=rng,
                        shuffle_on_cpu=shuffle_on_cpu,
                    )
                    fill = 0

        del data
        gc.collect()

    if fill > 0:
        yield make_host_batch_from_buffer(
            x_buf=x_buf,
            y_buf=y_buf,
            n_rows=fill,
            rng=rng,
            shuffle_on_cpu=shuffle_on_cpu,
        )


def preload_in_memory_cache(
    train_files: list[str],
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    cache_dtype: np.dtype,
) -> tuple[list[np.ndarray], np.ndarray, int]:
    cached_hiddens: list[np.ndarray] = []
    cached_labels: list[int] = []
    cached_positions = 0

    for pf in tqdm(train_files, desc="Preloading train cache"):
        path = Path(pf)
        if not path.exists():
            print(f"  WARNING: missing train file: {path}")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        for prot in data:
            sample = extract_sample(prot, tid_to_class_idx)
            if sample is None:
                continue
            hidden, cls_idx = sample

            if hidden.shape[1] != hidden_dim:
                raise ValueError(
                    f"Inconsistent hidden dim in {path}: "
                    f"{hidden.shape[1]} vs expected {hidden_dim}"
                )

            hidden = np.asarray(hidden, dtype=cache_dtype, order="C")
            cached_hiddens.append(hidden)
            cached_labels.append(int(cls_idx))
            cached_positions += int(hidden.shape[0])

        del data
        gc.collect()

    labels_np = np.asarray(cached_labels, dtype=np.int64)
    if len(cached_hiddens) == 0:
        raise RuntimeError("In-memory preload found zero valid proteins.")

    return cached_hiddens, labels_np, cached_positions


def iter_train_batches_in_memory(
    cached_hiddens: list[np.ndarray],
    cached_labels: np.ndarray,
    hidden_dim: int,
    batch_size_positions: int,
    rng: np.random.Generator,
    epoch_idx: int,
    shuffle_on_cpu: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    order = rng.permutation(len(cached_hiddens))
    print(f"Epoch {epoch_idx + 1} cached protein shuffle preview: {order[:8].tolist()}")

    x_buf = np.empty((batch_size_positions, hidden_dim), dtype=np.float32)
    y_buf = np.empty((batch_size_positions,), dtype=np.int64)
    fill = 0

    for protein_idx in order:
        hidden = cached_hiddens[int(protein_idx)]
        cls_idx = int(cached_labels[int(protein_idx)])

        start = 0
        total = hidden.shape[0]
        while start < total:
            room = batch_size_positions - fill
            take = min(room, total - start)
            x_buf[fill : fill + take] = hidden[start : start + take]
            y_buf[fill : fill + take] = cls_idx
            fill += take
            start += take

            if fill == batch_size_positions:
                yield make_host_batch_from_buffer(
                    x_buf=x_buf,
                    y_buf=y_buf,
                    n_rows=fill,
                    rng=rng,
                    shuffle_on_cpu=shuffle_on_cpu,
                )
                fill = 0

    if fill > 0:
        yield make_host_batch_from_buffer(
            x_buf=x_buf,
            y_buf=y_buf,
            n_rows=fill,
            rng=rng,
            shuffle_on_cpu=shuffle_on_cpu,
        )


def iter_train_batches_mmap(
    cache_dir: Path,
    cache_index: dict[str, Any],
    hidden_dim: int,
    batch_size_positions: int,
    rng: np.random.Generator,
    epoch_idx: int,
    shuffle_mode: str,
    shuffle_block_rows: int,
    host_dtype: np.dtype,
    shuffle_on_cpu: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if shuffle_block_rows <= 0:
        raise ValueError("--shuffle_block_rows must be > 0")

    shards = list(cache_index["shards"])
    if len(shards) == 0:
        raise RuntimeError("mmap cache has no shards")

    cache_dtype = cache_dtype_from_index(cache_index)
    shard_order = np.arange(len(shards), dtype=np.int64)
    if shuffle_mode in ("chunk", "global") and len(shards) > 1:
        shard_order = rng.permutation(len(shards))
    shard_preview = [str(shards[int(i)]["x_file"]) for i in shard_order[:5]]
    print(f"Epoch {epoch_idx + 1} mmap shard order preview: {shard_preview}")

    x_buf = np.empty((batch_size_positions, hidden_dim), dtype=host_dtype)
    y_buf = np.empty((batch_size_positions,), dtype=np.int64)
    fill = 0

    for shard_i in tqdm(shard_order, desc=f"Epoch {epoch_idx + 1} mmap shards"):
        shard = shards[int(shard_i)]
        rows = int(shard["rows"])
        x_path = cache_dir / str(shard["x_file"])
        y_path = cache_dir / str(shard["y_file"])
        # x_mm = np.memmap(
        #     x_path,
        #     mode="r",
        #     dtype=cache_dtype,
        #     shape=(rows, hidden_dim),
        # )
        # y_mm = np.memmap(
        #     y_path,
        #     mode="r",
        #     dtype=np.int32,
        #     shape=(rows,),
        # )
        x_mm = np.fromfile(x_path, dtype=cache_dtype).reshape(rows, hidden_dim)
        y_mm = np.fromfile(y_path, dtype=np.int32)

        if shuffle_mode == "sequential":
            block_starts = np.arange(0, rows, shuffle_block_rows, dtype=np.int64)
            for block_start in block_starts:
                start = int(block_start)
                end = min(rows, start + shuffle_block_rows)
                x_rows = np.asarray(x_mm[start:end], dtype=host_dtype)
                y_rows = np.asarray(y_mm[start:end], dtype=np.int64)

                row_start = 0
                total = int(x_rows.shape[0])
                while row_start < total:
                    room = batch_size_positions - fill
                    take = min(room, total - row_start)
                    x_buf[fill : fill + take] = x_rows[row_start : row_start + take]
                    y_buf[fill : fill + take] = y_rows[row_start : row_start + take]
                    fill += take
                    row_start += take
                    if fill == batch_size_positions:
                        yield make_host_batch_from_buffer(
                            x_buf=x_buf,
                            y_buf=y_buf,
                            n_rows=fill,
                            rng=rng,
                            shuffle_on_cpu=shuffle_on_cpu,
                        )
                        fill = 0

        elif shuffle_mode == "global":
            row_order = rng.permutation(rows)
            for start in range(0, rows, shuffle_block_rows):
                end = min(rows, start + shuffle_block_rows)
                idx = row_order[start:end]
                x_rows = np.asarray(x_mm[idx], dtype=host_dtype)
                y_rows = np.asarray(y_mm[idx], dtype=np.int64)

                row_start = 0
                total = int(x_rows.shape[0])
                while row_start < total:
                    room = batch_size_positions - fill
                    take = min(room, total - row_start)
                    x_buf[fill : fill + take] = x_rows[row_start : row_start + take]
                    y_buf[fill : fill + take] = y_rows[row_start : row_start + take]
                    fill += take
                    row_start += take
                    if fill == batch_size_positions:
                        yield make_host_batch_from_buffer(
                            x_buf=x_buf,
                            y_buf=y_buf,
                            n_rows=fill,
                            rng=rng,
                            shuffle_on_cpu=shuffle_on_cpu,
                        )
                        fill = 0

        else:  # shuffle_mode == "chunk"
            block_starts = np.arange(0, rows, shuffle_block_rows, dtype=np.int64)
            if len(block_starts) > 1:
                rng.shuffle(block_starts)
            for block_start in block_starts:
                start = int(block_start)
                end = min(rows, start + shuffle_block_rows)
                block_len = end - start
                if block_len <= 0:
                    continue
                if block_len > 1:
                    block_perm = rng.permutation(block_len)
                    x_rows = np.asarray(x_mm[start:end][block_perm], dtype=host_dtype)
                    y_rows = np.asarray(y_mm[start:end][block_perm], dtype=np.int64)
                else:
                    x_rows = np.asarray(x_mm[start:end], dtype=host_dtype)
                    y_rows = np.asarray(y_mm[start:end], dtype=np.int64)

                row_start = 0
                total = int(x_rows.shape[0])
                while row_start < total:
                    room = batch_size_positions - fill
                    take = min(room, total - row_start)
                    x_buf[fill : fill + take] = x_rows[row_start : row_start + take]
                    y_buf[fill : fill + take] = y_rows[row_start : row_start + take]
                    fill += take
                    row_start += take
                    if fill == batch_size_positions:
                        yield make_host_batch_from_buffer(
                            x_buf=x_buf,
                            y_buf=y_buf,
                            n_rows=fill,
                            rng=rng,
                            shuffle_on_cpu=shuffle_on_cpu,
                        )
                        fill = 0

        del x_mm
        del y_mm
        # _drop_page_cache(x_path)
        # _drop_page_cache(y_path)

    if fill > 0:
        yield make_host_batch_from_buffer(
            x_buf=x_buf,
            y_buf=y_buf,
            n_rows=fill,
            rng=rng,
            shuffle_on_cpu=shuffle_on_cpu,
        )


@torch.no_grad()
def evaluate(
    files: list[str],
    model: nn.Module,
    tid_to_class_idx: dict[int, int],
    hidden_dim: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray, int]:
    model.eval()
    n_cls = int(model.out_features)  # type: ignore[attr-defined]
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)
    n_proteins = 0

    for pf in files:
        path = Path(pf)
        if not path.exists():
            print(f"  WARNING: missing eval file: {path}")
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)

        for prot in data:
            sample = extract_sample(prot, tid_to_class_idx)
            if sample is None:
                continue
            hidden, cls_idx = sample

            if hidden.shape[1] != hidden_dim:
                raise ValueError(
                    f"Inconsistent hidden dim in {path}: "
                    f"{hidden.shape[1]} vs expected {hidden_dim}"
                )

            x = torch.from_numpy(np.asarray(hidden, dtype=np.float32)).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            preds = model(x).argmax(dim=1).cpu().numpy()
            conf[cls_idx] += np.bincount(preds, minlength=n_cls)
            n_proteins += 1

        del data
        gc.collect()

    total = conf.sum()
    overall_acc = float(conf.trace() / total) if total > 0 else 0.0

    class_totals = conf.sum(axis=1)
    valid = class_totals > 0
    per_class_acc = np.zeros(n_cls, dtype=np.float64)
    per_class_acc[valid] = conf.diagonal()[valid] / class_totals[valid]
    bal_acc = float(per_class_acc[valid].mean()) if valid.any() else 0.0

    return bal_acc, overall_acc, conf, n_proteins


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    best_val_bacc: float,
    history: list[dict],
    args: argparse.Namespace,
    data_mode_actual: str,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "best_val_bacc": best_val_bacc,
        "history": history,
        "args": vars(args),
        "data_mode_actual": data_mode_actual,
    }
    torch.save(checkpoint, path)


def run() -> None:
    args = parse_args()
    if args.eval_every <= 0:
        raise ValueError("--eval_every must be >= 1")
    if args.prefetch_batches < 0:
        raise ValueError("--prefetch_batches must be >= 0")
    if args.disk_prefetch_batches < 0:
        raise ValueError("--disk_prefetch_batches must be >= 0")
    if args.build_cache_only:
        args.build_cache = True

    set_seed(args.seed)
    device = resolve_device(args.device)
    shuffle_device_actual = args.shuffle_device
    if shuffle_device_actual == "cuda" and device.type != "cuda":
        print("shuffle_device=cuda requested without CUDA device; falling back to cpu.")
        shuffle_device_actual = "cpu"
    shuffle_on_cuda = shuffle_device_actual == "cuda" and device.type == "cuda"
    shuffle_on_cpu = not shuffle_on_cuda

    host_dtype_np = np.float16 if args.host_dtype == "float16" else np.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # write args to output dir for reproducibility
    save_run_metadata(output_dir, args)

    train_files = collect_train_files(args.train_glob, args.max_train_files)
    val_files = parse_csv_paths(args.val_files)
    test_files = parse_csv_paths(args.test_files)

    rank_mapping = load_rank_mapping(args.taxonomy_mapping_file, args.rank)
    valid_classes, protein_counts, position_counts, hidden_dim = build_or_load_counts(
        output_dir=output_dir,
        rank=args.rank,
        min_class_count=args.min_class_count,
        train_files=train_files,
        rank_mapping=rank_mapping,
        count_num_workers=args.cache_num_workers,
        count_start_method=args.cache_start_method,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(valid_classes))
    classes = label_encoder.classes_
    n_cls = len(classes)
    label_to_idx = {int(label): idx for idx, label in enumerate(classes)}

    # Restrict to only species IDs mapping to valid classes
    tid_to_class_idx = {
        int(tid): label_to_idx[int(label)]
        for tid, label in rank_mapping.items()
        if int(label) in label_to_idx
    }

    pos_per_class = np.array(
        [position_counts[int(c)] for c in classes],
        dtype=np.int64,
    )
    if args.class_weight_mode == "balanced":
        total_positions = int(pos_per_class.sum())
        class_weights = np.zeros(n_cls, dtype=np.float64)
        np.divide(
            float(total_positions),
            n_cls * pos_per_class.astype(np.float64),
            out=class_weights,
            where=pos_per_class > 0,
        )
    elif args.class_weight_mode == "log":
        total_positions = int(pos_per_class.sum())
        class_weights = np.log1p(
            float(total_positions) / (n_cls * pos_per_class.astype(np.float64))
        )
        class_weights /= class_weights.mean()
    elif args.class_weight_mode == "effective":
        beta = 1.0 - 1.0 / np.median(pos_per_class)
        effective_n = (1.0 - beta ** pos_per_class.astype(np.float64)) / (1.0 - beta)
        class_weights = 1.0 / effective_n
        class_weights *= len(class_weights) / class_weights.sum()
    elif args.class_weight_mode == "none":
        class_weights = None
    else:
        raise ValueError(f"Unknown class_weight_mode: {args.class_weight_mode}")
    print(
        f"Training rank={args.rank}, classes={n_cls}, hidden_dim={hidden_dim}, "
        f"device={device}, amp={bool(args.amp and device.type == 'cuda')}, "
        f"shuffle_device={shuffle_device_actual}, prefetch_batches={args.prefetch_batches}, "
        f"disk_prefetch_batches={args.disk_prefetch_batches}, host_dtype={args.host_dtype}"
    )

    cache_dtype_np = np.float16 if args.cache_dtype == "float16" else np.float32
    cache_dir = resolve_cache_dir(
        cache_dir_arg=args.cache_dir,
        output_dir=output_dir,
        rank=args.rank,
        min_class_count=args.min_class_count,
        cache_dtype=args.cache_dtype,
    )
    expected_class_hash = class_mapping_hash(
        classes=classes, tid_to_class_idx=tid_to_class_idx
    )
    cache_index: dict[str, Any] | None = None
    cache_lookup_requested = (
        args.build_cache
        or args.build_cache_only
        or args.data_mode
        in (
            "mmap",
            "auto",
        )
    )
    if cache_lookup_requested:
        try:
            cache_index = build_or_load_mmap_cache(
                train_files=train_files,
                tid_to_class_idx=tid_to_class_idx,
                hidden_dim=hidden_dim,
                n_cls=n_cls,
                classes=classes,
                expected_class_hist=pos_per_class,
                expected_class_hash=expected_class_hash,
                cache_dir=cache_dir,
                cache_dtype=cache_dtype_np,
                mmap_shard_rows=args.mmap_shard_rows,
                cache_num_workers=args.cache_num_workers,
                cache_start_method=args.cache_start_method,
                rank=args.rank,
                min_class_count=args.min_class_count,
                force_rebuild=args.build_cache,
                build_if_missing=(
                    args.build_cache
                    or args.build_cache_only
                    or args.data_mode == "mmap"
                ),
            )
        except FileNotFoundError:
            if args.build_cache or args.build_cache_only or args.data_mode == "mmap":
                raise
            if args.data_mode == "auto":
                print(
                    f"No mmap cache found in {cache_dir}; "
                    "auto mode will continue without mmap fallback."
                )
            else:
                raise

    if args.build_cache_only:
        if cache_index is None:
            raise RuntimeError("Requested --build_cache_only but cache was not built.")
        print(f"Cache build completed at {cache_dir}. Exiting (--build_cache_only).")
        return

    model = nn.Linear(hidden_dim, n_cls).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=(
            torch.tensor(class_weights, dtype=torch.float32, device=device)
            if class_weights is not None
            else None
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.lr_scheduler == "reduce_on_plateau":
        print(
            f"Using ReduceLROnPlateau scheduler with factor={args.lr_scheduler_factor}"
            f"patience={args.lr_scheduler_patience}, min_lr={args.lr_scheduler_min_lr}"
            f"threshold={args.lr_scheduler_threshold}"
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
            threshold=args.lr_scheduler_threshold,
        )

    amp_enabled = bool(args.amp and device.type == "cuda")
    x_device_dtype = (
        torch.float16
        if (amp_enabled and host_dtype_np == np.float16)
        else torch.float32
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    data_mode_actual = args.data_mode
    cached_hiddens: list[np.ndarray] | None = None
    cached_labels: np.ndarray | None = None
    if args.data_mode == "mmap":
        if cache_index is None:
            raise RuntimeError(
                "mmap mode requested but no valid cache index is available. "
                "Use --build_cache or provide --cache_dir with an existing cache."
            )
        data_mode_actual = "mmap"
    elif args.data_mode in ("in_memory", "auto"):
        dtype_bytes = np.dtype(cache_dtype_np).itemsize
        estimated_hidden_bytes = (
            int(total_positions) * int(hidden_dim) * int(dtype_bytes)
        )
        estimated_total_bytes = int(estimated_hidden_bytes * 1.15)
        print(
            "Estimated in-memory cache (hidden only): "
            f"{estimated_hidden_bytes / (1024**3):.2f} GiB; "
            f"with overhead ~{estimated_total_bytes / (1024**3):.2f} GiB"
        )

        try:
            t0 = time.time()
            cached_hiddens, cached_labels, cached_positions = preload_in_memory_cache(
                train_files=train_files,
                tid_to_class_idx=tid_to_class_idx,
                hidden_dim=hidden_dim,
                cache_dtype=cache_dtype_np,
            )
            actual_bytes = int(sum(arr.nbytes for arr in cached_hiddens))
            print(
                f"Loaded in-memory cache: {len(cached_hiddens):,} proteins, "
                f"{cached_positions:,} positions, "
                f"{actual_bytes / (1024**3):.2f} GiB raw arrays, "
                f"{time.time() - t0:.1f}s"
            )
            data_mode_actual = "in_memory"
        except MemoryError as exc:
            if args.data_mode == "in_memory":
                raise
            if cache_index is not None:
                print(
                    f"In-memory preload failed ({exc.__class__.__name__}): "
                    "falling back to mmap mode."
                )
                data_mode_actual = "mmap"
            else:
                print(
                    f"In-memory preload failed ({exc.__class__.__name__}): "
                    "falling back to stream mode."
                )
                data_mode_actual = "stream"
            cached_hiddens = None
            cached_labels = None
    else:
        data_mode_actual = "stream"

    print(f"Data mode requested={args.data_mode}, actual={data_mode_actual}")

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
        print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_bacc = float(ckpt.get("best_val_bacc", -1.0))
        history = list(ckpt.get("history", []))
        print(f"Resumed at epoch {start_epoch}, best_val_bacc={best_val_bacc:.4f}")

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if checkpoint_best.exists():
        best_ckpt = torch.load(checkpoint_best, map_location="cpu")
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in best_ckpt["model_state_dict"].items()
        }
        best_val_bacc = max(best_val_bacc, float(best_ckpt.get("best_val_bacc", -1.0)))
    if best_val_bacc < 0:
        best_val_bacc = -1.0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        epoch_rng = np.random.default_rng(args.seed + epoch)
        cuda_shuffle_generator: torch.Generator | None = None
        if shuffle_on_cuda:
            cuda_shuffle_generator = torch.Generator(device=device)
            cuda_shuffle_generator.manual_seed(int(args.seed + epoch))

        model.train()
        epoch_loss = 0.0
        epoch_positions = 0
        n_batches = 0
        prev_lr = float(optimizer.param_groups[0]["lr"])

        if data_mode_actual == "in_memory":
            assert cached_hiddens is not None and cached_labels is not None
            host_batch_iter = iter_train_batches_in_memory(
                cached_hiddens=cached_hiddens,
                cached_labels=cached_labels,
                hidden_dim=hidden_dim,
                batch_size_positions=args.batch_size_positions,
                rng=epoch_rng,
                epoch_idx=epoch,
                shuffle_on_cpu=shuffle_on_cpu,
            )
        elif data_mode_actual == "mmap":
            assert cache_index is not None
            host_batch_iter = iter_train_batches_mmap(
                cache_dir=cache_dir,
                cache_index=cache_index,
                hidden_dim=hidden_dim,
                batch_size_positions=args.batch_size_positions,
                rng=epoch_rng,
                epoch_idx=epoch,
                shuffle_mode=args.shuffle_mode,
                shuffle_block_rows=args.shuffle_block_rows,
                host_dtype=host_dtype_np,
                shuffle_on_cpu=shuffle_on_cpu,
            )
        else:
            host_batch_iter = iter_train_batches_stream(
                train_files=train_files,
                tid_to_class_idx=tid_to_class_idx,
                hidden_dim=hidden_dim,
                batch_size_positions=args.batch_size_positions,
                rng=epoch_rng,
                epoch_idx=epoch,
                shuffle_on_cpu=shuffle_on_cpu,
            )

        host_batch_iter = iter_host_batches_prefetch(
            host_batch_iter=host_batch_iter,
            disk_prefetch_batches=args.disk_prefetch_batches,
        )
        device_batch_iter = iter_device_batches_prefetch(
            host_batch_iter=host_batch_iter,
            device=device,
            x_device_dtype=x_device_dtype,
            shuffle_on_cuda=shuffle_on_cuda,
            cuda_shuffle_generator=cuda_shuffle_generator,
            prefetch_batches=args.prefetch_batches,
        )
        for x_batch, y_batch in device_batch_iter:
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                prev_lr = float(optimizer.param_groups[0]["lr"])
                scheduler.step(float(loss.item()))
                new_lr = float(optimizer.param_groups[0]["lr"])
                if new_lr < prev_lr:
                    print(
                        f"  [batch {n_batches}] LR reduced: {prev_lr:.3e} -> {new_lr:.3e} "
                        f"(loss={loss.item():.4f})"
                    )

            batch_positions = int(x_batch.shape[0])
            epoch_loss += float(loss.item()) * batch_positions
            epoch_positions += batch_positions
            n_batches += 1

            if n_batches % 10 == 0 or n_batches == 1:
                print(
                    f"Epoch {epoch + 1} progress: "
                    f"{epoch_positions:,} positions, {n_batches:,} batches, "
                    f"current_loss={float(loss.item()):.4f}"
                )

        train_loss = epoch_loss / max(epoch_positions, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])

        should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epochs - 1)
        improved = False
        val_bacc: float | None = None
        val_acc: float | None = None
        val_n = 0
        if should_eval:
            val_bacc, val_acc, _, val_n = evaluate(
                files=val_files,
                model=model,
                tid_to_class_idx=tid_to_class_idx,
                hidden_dim=hidden_dim,
                device=device,
            )
            improved = float(val_bacc) > best_val_bacc
            if improved:
                best_val_bacc = float(val_bacc)
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_positions": epoch_positions,
            "train_batches": n_batches,
            "val_balanced_acc": (float(val_bacc) if val_bacc is not None else None),
            "val_overall_acc": (float(val_acc) if val_acc is not None else None),
            "val_n_proteins": val_n,
            "best_val_balanced_acc": best_val_bacc,
            "lr": current_lr,
            "epoch_time_sec": time.time() - t0,
        }
        history.append(epoch_metrics)

        if should_eval:
            assert val_bacc is not None and val_acc is not None
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_bacc={val_bacc:.4f} | val_acc={val_acc:.4f} | "
                f"lr={current_lr:.3e} | "
                f"positions={epoch_positions:,} | batches={n_batches:,} | "
                f"time={epoch_metrics['epoch_time_sec']:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"loss={train_loss:.4f} | "
                f"val=skipped(eval_every={args.eval_every}) | "
                f"lr={current_lr:.3e} | "
                f"positions={epoch_positions:,} | batches={n_batches:,} | "
                f"time={epoch_metrics['epoch_time_sec']:.1f}s"
            )
        if scheduler is not None and current_lr < prev_lr:
            print(
                f"  LR reduced by scheduler on train step loss: "
                f"{prev_lr:.3e} -> {current_lr:.3e}"
            )

        save_checkpoint(
            path=checkpoint_last,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_bacc=best_val_bacc,
            history=history,
            args=args,
            data_mode_actual=data_mode_actual,
        )
        if improved:
            save_checkpoint(
                path=checkpoint_best,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_bacc=best_val_bacc,
                history=history,
                args=args,
                data_mode_actual=data_mode_actual,
            )
        if args.save_every_epoch:
            save_checkpoint(
                path=output_dir / f"checkpoint_epoch_{epoch}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_bacc=best_val_bacc,
                history=history,
                args=args,
                data_mode_actual=data_mode_actual,
            )

    # Restore best epoch weights before final evaluation/export.
    model.load_state_dict(best_state)

    val_bacc, val_acc, val_conf, val_n = evaluate(
        files=val_files,
        model=model,
        tid_to_class_idx=tid_to_class_idx,
        hidden_dim=hidden_dim,
        device=device,
    )
    test_bacc, test_acc, test_conf, test_n = evaluate(
        files=test_files,
        model=model,
        tid_to_class_idx=tid_to_class_idx,
        hidden_dim=hidden_dim,
        device=device,
    )

    # Save model weights
    model_path = output_dir / f"probe_{args.rank}_model.pt"
    torch.save(model.state_dict(), model_path)

    with torch.no_grad():
        w = model.weight.detach().cpu().numpy().T.astype(np.float64)  # [d, n_cls]
        intercept = model.bias.detach().cpu().numpy().astype(np.float64)  # [n_cls]

    probe_data = {
        "W": w,
        "intercept": intercept,
        "classes": classes,
        "class_weights": class_weights,
        "pos_per_class": pos_per_class,
        "total_positions": int(pos_per_class.sum()),
        "n_classes": int(n_cls),
        "rank_mapping": rank_mapping,
        "rank": args.rank,
        "data_mode_requested": args.data_mode,
        "data_mode_actual": data_mode_actual,
        "cache_dir": str(cache_dir) if cache_index is not None else None,
    }
    probe_data_path = output_dir / f"probe_{args.rank}_data.pkl"
    with open(probe_data_path, "wb") as f:
        pickle.dump(probe_data, f)

    metrics = {
        "rank": args.rank,
        "n_classes": int(n_cls),
        "hidden_dim": int(hidden_dim),
        "chance": float(1.0 / n_cls),
        "best_val_balanced_acc": float(best_val_bacc),
        "final_val_balanced_acc": float(val_bacc),
        "final_val_overall_acc": float(val_acc),
        "final_val_n_proteins": int(val_n),
        "final_test_balanced_acc": float(test_bacc),
        "final_test_overall_acc": float(test_acc),
        "final_test_n_proteins": int(test_n),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "lr_scheduler_threshold": args.lr_scheduler_threshold,
        "data_mode_requested": args.data_mode,
        "data_mode_actual": data_mode_actual,
        "cache_dtype": args.cache_dtype,
        "cache_dir": str(cache_dir),
        "cache_total_rows": (
            int(cache_index["total_rows"]) if cache_index is not None else None
        ),
        "cache_build_num_workers": (
            int(cache_index.get("build_num_workers", 1))
            if cache_index is not None
            else None
        ),
        "cache_build_strategy": (
            str(cache_index.get("build_strategy")) if cache_index is not None else None
        ),
        "mmap_shard_rows": args.mmap_shard_rows,
        "cache_num_workers": args.cache_num_workers,
        "cache_start_method": args.cache_start_method,
        "shuffle_mode": args.shuffle_mode,
        "shuffle_device": shuffle_device_actual,
        "shuffle_block_rows": args.shuffle_block_rows,
        "prefetch_batches": args.prefetch_batches,
        "disk_prefetch_batches": args.disk_prefetch_batches,
        "host_dtype": args.host_dtype,
        "eval_every": args.eval_every,
        "history": history,
    }
    metrics_path = output_dir / f"metrics_{args.rank}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    conf_path = output_dir / f"confusion_{args.rank}.npz"
    np.savez(conf_path, val=val_conf, test=test_conf, classes=classes)

    print(
        f"Saved artifacts:\n"
        f"  {model_path}\n"
        f"  {probe_data_path}\n"
        f"  {metrics_path}\n"
        f"  {conf_path}\n"
        f"  {checkpoint_last}\n"
        f"  {checkpoint_best}"
    )

    # Clean up mmap cache
    if cache_index is not None and cache_dir.exists():
        removed = cleanup_cache_shards_from_index(cache_dir, cache_index)
        index_path = cache_dir / "cache_index.json"
        index_path.unlink(missing_ok=True)
        try:
            cache_dir.rmdir()  # only succeeds if empty
        except OSError:
            pass
        print(f"Cleaned up mmap cache: removed {removed} shard files from {cache_dir}")


if __name__ == "__main__":
    run()

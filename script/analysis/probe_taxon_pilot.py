# mypy: ignore-errors
"""
Pilot study: Layer-wise linear probing for taxonomic information
in Dayhoff hidden states.

Trains L2-regularized logistic regression probes at each layer to predict
taxonomic rank (domain, phylum, class, order, ...) from hidden representations.

Two modes:
  - Pooled: mean-pool hidden states across positions -> one vector per protein
  - Per-position: individual position hidden states -> one vector per residue
    (with protein-level CV splits to prevent leakage)

Reports balanced accuracy per layer for both modes, plus the gap.

Requirements:
  pip install numpy scikit-learn matplotlib torch

Memory optimization:
  - Pooled features extracted once per layer (tiny: N × 1280 float16)
  - Per-position features extracted per-rank, directly in float32,
    only for that rank's proteins. Freed before next rank.
  - No full per_pos array for all proteins ever exists.
  Peak ≈ pickle_size + largest_single_rank_perpos (~7-8 GB at 1024 positions)
"""

import pickle
import gc
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# CONFIGURATION — edit these
# ═══════════════════════════════════════════════

PICKLE_FILES = [
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_last_protein_hidden_train_0.pkl",  # each contains a list of protein dicts
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_last_protein_hidden_train_1.pkl",
]

# Precomputed taxonomy mapping:
# taxid_to_std_ranks[species_taxid] = {'domain': taxid, 'phylum': taxid, ...}
TAXONOMY_MAPPING_FILE = "/scratch/suyuelyu/deimm/data/oma/taxid_to_std_ranks.pkl"

# Taxonomic ranks to probe (coarse to fine)
# Must match keys in taxid_to_std_ranks values
RANKS_TO_PROBE = ["domain", "phylum", "class", "order", "family", "genus"]

# Probe hyperparameters
N_FOLDS = 5
C_REG = 1.0  # inverse regularization strength (lower = stronger L2)
MAX_ITER = 2000
MIN_CLASS_COUNT = 10  # drop classes with fewer proteins than this

# Layer indexing: 33 items, index 0 = embedding layer, 1-32 = transformer layers
N_LAYERS = 33

# Set to a list like [0, 4, 8, 12, 16, 20, 24, 28, 32] to probe subset
# Set to None to probe all 33 layers
LAYERS_TO_PROBE = [0, 8, 16, 24, 32]

# Per-position memory control: subsample positions per protein
# Set None to use all positions (uses more memory)
# Set e.g. 50 to randomly sample 50 positions per protein
MAX_POSITIONS_PER_PROTEIN = None

# Output
OUTPUT_DIR = Path("/scratch/suyuelyu/deimm/results/probe_taxon/pilot_per8lyr_results")
RANDOM_SEED = 42


# ═══════════════════════════════════════════════
# 1. TAXONOMY MAPPING (from precomputed dict)
# ═══════════════════════════════════════════════


def load_taxonomy_mapping(mapping_file, ranks):
    """
    Load precomputed taxid_to_std_ranks and reshape into per-rank lookups.

    Input format:  {species_taxid: {'domain': rank_taxid, 'phylum': rank_taxid, ...}}
    Output format: {rank_name: {species_taxid: rank_taxid_as_label}}

    The rank taxids are used directly as class labels (integers).
    """
    print(f"Loading taxonomy mapping from {mapping_file}...")
    with open(mapping_file, "rb") as f:
        taxid_to_std_ranks = pickle.load(f)

    rank_mapping = {r: {} for r in ranks}
    for species_tid, rank_dict in taxid_to_std_ranks.items():
        for rank in ranks:
            if rank in rank_dict:
                rank_mapping[rank][int(species_tid)] = int(rank_dict[rank])

    for rank in ranks:
        n_mapped = len(rank_mapping[rank])
        n_labels = len(set(rank_mapping[rank].values()))
        print(f"  {rank}: {n_mapped} species mapped, {n_labels} unique groups")

    return rank_mapping


# ═══════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════


def load_proteins(pickle_files):
    """Load all pickle files, return list of protein dicts."""
    all_proteins = []
    for pf in pickle_files:
        pf = Path(pf)
        if not pf.exists():
            print(f"WARNING: {pf} not found, skipping")
            continue
        print(f"Loading {pf}...")
        with open(pf, "rb") as f:
            data = pickle.load(f)
        print(f"  {len(data)} proteins loaded")
        all_proteins.extend(data)
    print(f"Total: {len(all_proteins)} proteins\n")
    return all_proteins


# ═══════════════════════════════════════════════
# 3. FEATURE EXTRACTION
# ═══════════════════════════════════════════════


def extract_pooled_for_layer(proteins, layer_idx):
    """
    Extract mean-pooled hidden states for ONE layer across all proteins.
    Returns: np.float16 [n_proteins, 1280]  (tiny array)
    """
    pooled_list = []
    for prot in proteins:
        h = prot["last_protein_hiddens"][layer_idx]
        if hasattr(h, "numpy"):
            h = h.numpy()
        # Mean in float32 for numerical stability, store as float16
        pooled_list.append(h.astype(np.float32).mean(axis=0).astype(np.float16))
    return np.stack(pooled_list)


def extract_perposition_for_rank(proteins, layer_idx, indices, max_pos=None, seed=42):
    """
    Extract per-position hidden states for ONE layer, only for proteins
    at the given indices. Returns float32 directly (no intermediate float16).

    Args:
        proteins:   list of protein dicts
        layer_idx:  which layer
        indices:    list of protein indices to include
        max_pos:    if set, subsample this many positions per protein
        seed:       random seed for subsampling

    Returns:
        X_pp:       np.float32 [n_total_positions, 1280]
        prot_idx:   np.int32   [n_total_positions] — consecutive index (0..len(indices)-1)
    """
    rng = np.random.RandomState(seed)
    pp_list = []
    pp_prot = []

    for new_idx, old_idx in enumerate(indices):
        h = proteins[old_idx]["last_protein_hiddens"][layer_idx]
        if hasattr(h, "numpy"):
            h = h.numpy()
        h = h.astype(np.float32)

        seq_len = h.shape[0]
        if max_pos is not None and seq_len > max_pos:
            sel = rng.choice(seq_len, max_pos, replace=False)
            sel.sort()
            h = h[sel]

        pp_list.append(h)
        pp_prot.append(np.full(h.shape[0], new_idx, dtype=np.int32))

    X_pp = np.concatenate(pp_list)
    prot_idx = np.concatenate(pp_prot)
    del pp_list
    return X_pp, prot_idx


# ═══════════════════════════════════════════════
# 4. PROBE TRAINING
# ═══════════════════════════════════════════════


def probe_pooled(X, labels, n_folds, C):
    """
    L2-regularized logistic regression on pooled features.
    Stratified k-fold CV. Returns (mean_balanced_acc, n_classes).
    X may be float16 — upcast to float32 for sklearn.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_cls = len(le.classes_)
    if n_cls < 2:
        return np.nan, n_cls

    X = X.astype(np.float32) if X.dtype != np.float32 else X

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    accs = []

    for tr, va in skf.split(X, y):
        clf = LogisticRegression(
            C=C,
            max_iter=MAX_ITER,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[va], clf.predict(X[va])))

    return float(np.mean(accs)), n_cls


# def probe_perposition(X_pp, prot_idx, labels_protein, n_folds, C):
#     """
#     L2-regularized SGD logistic regression on per-position features.
#     CV splits are at the PROTEIN level (not position level) to prevent
#     leakage from the same protein appearing in train and val.

#     X_pp is expected to be float32 already (from extract_perposition_for_rank).

#     Args:
#         X_pp:            [n_positions, 1280] float32
#         prot_idx:        [n_positions] — consecutive protein index (0..n_proteins-1)
#         labels_protein:  list of length n_proteins — label per protein
#         n_folds, C:      hyperparams
#     """
#     le = LabelEncoder()
#     le.fit(labels_protein)
#     labels_arr = np.array(labels_protein)
#     y_pos = le.transform(labels_arr[prot_idx])
#     y_prot = le.transform(labels_protein)
#     n_cls = len(le.classes_)
#     if n_cls < 2:
#         return np.nan, n_cls

#     n_prot = len(labels_protein)
#     skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
#     accs = []

#     # SGD alpha ≈ 1 / (C * n_samples)
#     alpha = 1.0 / (C * len(y_pos))

#     for tr_prot, va_prot in skf.split(np.arange(n_prot), y_prot):
#         tr_set = set(tr_prot.tolist())
#         va_set = set(va_prot.tolist())

#         tr_mask = np.array([p in tr_set for p in prot_idx])
#         va_mask = np.array([p in va_set for p in prot_idx])

#         clf = SGDClassifier(
#             loss="log_loss",
#             alpha=alpha,
#             max_iter=MAX_ITER,
#             class_weight="balanced",
#             random_state=RANDOM_SEED,
#             tol=1e-3,
#         )
#         clf.fit(X_pp[tr_mask], y_pos[tr_mask])
#         pred = clf.predict(X_pp[va_mask])
#         accs.append(balanced_accuracy_score(y_pos[va_mask], pred))


#     return float(np.mean(accs)), n_cls
def probe_perposition(X_pp, prot_idx, labels_protein, n_folds, C):
    le = LabelEncoder()
    le.fit(labels_protein)
    labels_arr = np.array(labels_protein)
    y_pos = le.transform(labels_arr[prot_idx])
    y_prot = le.transform(labels_protein)
    n_cls = len(le.classes_)
    if n_cls < 2:
        return np.nan, n_cls

    n_prot = len(labels_protein)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    accs = []

    for tr_prot, va_prot in skf.split(np.arange(n_prot), y_prot):
        tr_set = set(tr_prot.tolist())
        va_set = set(va_prot.tolist())

        tr_mask = np.array([p in tr_set for p in prot_idx])
        va_mask = np.array([p in va_set for p in prot_idx])

        clf = RidgeClassifier(
            alpha=1.0 / C,
            class_weight="balanced",
        )
        clf.fit(X_pp[tr_mask], y_pos[tr_mask])
        pred = clf.predict(X_pp[va_mask])
        accs.append(balanced_accuracy_score(y_pos[va_mask], pred))

    return float(np.mean(accs)), n_cls


# ═══════════════════════════════════════════════
# 5. FILTER PROTEINS BY RANK
# ═══════════════════════════════════════════════


def filter_for_rank(taxids, rank_mapping, min_count):
    """
    Return indices and labels for proteins that:
    - have a valid mapping at this rank
    - belong to a class with at least min_count members

    Returns: (valid_indices, labels)  or (None, None) if insufficient data
    """
    idx_label = []
    for i, tid in enumerate(taxids):
        if tid in rank_mapping:
            idx_label.append((i, rank_mapping[tid]))

    if len(idx_label) < 50:
        return None, None

    # Filter rare classes
    labels_all = [lab for _, lab in idx_label]
    counts = Counter(labels_all)
    valid_classes = {c for c, n in counts.items() if n >= min_count}

    filtered = [(i, lab) for i, lab in idx_label if lab in valid_classes]
    if len(filtered) < 50 or len(valid_classes) < 2:
        return None, None

    indices = [i for i, _ in filtered]
    labels = [lab for _, lab in filtered]
    return indices, labels


# ═══════════════════════════════════════════════
# 6. MAIN PIPELINE
# ═══════════════════════════════════════════════


def run_pilot():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load data ──
    proteins = load_proteins(PICKLE_FILES)
    n_proteins = len(proteins)
    print(f"Total proteins loaded: {n_proteins}")
    taxids = [int(p["lin"]) for p in proteins]

    # ── Taxonomy mapping ──
    rank_mapping = load_taxonomy_mapping(TAXONOMY_MAPPING_FILE, RANKS_TO_PROBE)

    # ── Precompute valid indices per rank ──
    rank_indices = {}
    for rank in RANKS_TO_PROBE:
        indices, labels = filter_for_rank(taxids, rank_mapping[rank], MIN_CLASS_COUNT)
        if indices is not None:
            rank_indices[rank] = (indices, labels)
            n_cls = len(set(labels))
            print(f"  {rank}: {len(indices)} proteins, {n_cls} classes")
        else:
            print(f"  {rank}: SKIPPED (insufficient data)")

    active_ranks = list(rank_indices.keys())
    if not active_ranks:
        print("ERROR: no ranks have sufficient data. Check your taxonomy IDs.")
        return

    # ── Determine layers ──
    layers = LAYERS_TO_PROBE if LAYERS_TO_PROBE is not None else list(range(N_LAYERS))

    # ── Results storage ──
    results = {
        rank: {"layers": [], "pooled": [], "per_position": [], "n_classes": 0}
        for rank in active_ranks
    }
    for rank in active_ranks:
        results[rank]["n_classes"] = len(set(rank_indices[rank][1]))

    # ── Layer-by-layer probing ──
    for layer_idx in tqdm(layers):
        print(f"\n{'─' * 60}")
        print(f"Layer {layer_idx}")
        print(f"{'─' * 60}")

        # ── Extract pooled features once (tiny: N × 1280 float16) ──
        pooled = extract_pooled_for_layer(proteins, layer_idx)
        print(f"  pooled: {pooled.shape}")

        # ── Pass 1: pooled probes (cheap, shared array) ──
        for rank in active_ranks:
            indices, labels = rank_indices[rank]
            X_pool = pooled[indices]
            acc_pool, n_cls = probe_pooled(X_pool, labels, N_FOLDS, C_REG)
            results[rank]["layers"].append(layer_idx)
            results[rank]["pooled"].append(acc_pool)

        del pooled
        gc.collect()

        # ── Pass 2: per-position probes (one rank at a time) ──
        for rank in active_ranks:
            indices, labels = rank_indices[rank]
            n_cls = results[rank]["n_classes"]
            acc_pool = results[rank]["pooled"][-1]

            # Extract per-position only for this rank's proteins, in float32
            X_pp, prot_idx = extract_perposition_for_rank(
                proteins,
                layer_idx,
                indices,
                max_pos=MAX_POSITIONS_PER_PROTEIN,
                seed=RANDOM_SEED,
            )
            print(f"    {rank} per_pos: {X_pp.shape} ({X_pp.nbytes / 1e9:.1f} GB)")

            acc_pp, _ = probe_perposition(X_pp, prot_idx, labels, N_FOLDS, C_REG)

            del X_pp, prot_idx
            gc.collect()

            gap = (
                acc_pool - acc_pp
                if not (np.isnan(acc_pool) or np.isnan(acc_pp))
                else np.nan
            )
            print(
                f"  {rank:>15s}: pooled={acc_pool:.4f}  per_pos={acc_pp:.4f}  "
                f"gap={gap:+.4f}  ({n_cls} cls)"
            )
            results[rank]["per_position"].append(acc_pp)

    # ── Save and plot ──
    save_results(results)
    plot_results(results)
    print_summary(results)


# ═══════════════════════════════════════════════
# 7. OUTPUT
# ═══════════════════════════════════════════════


def save_results(results):
    """Save results as a pickle for later analysis."""
    out = OUTPUT_DIR / "probe_results.pkl"
    with open(out, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {out}")


def plot_results(results):
    """
    Two-panel plot per rank:
      Left:  pooled & per-position accuracy vs layer
      Right: gap (pooled - per_position) vs layer
    """
    ranks = list(results.keys())
    n_ranks = len(ranks)
    fig, axes = plt.subplots(n_ranks, 2, figsize=(14, 4 * n_ranks), squeeze=False)

    for i, rank in enumerate(ranks):
        data = results[rank]
        layers = np.array(data["layers"])
        pooled = np.array(data["pooled"])
        per_pos = np.array(data["per_position"])
        n_cls = data["n_classes"]
        chance = 1.0 / n_cls if n_cls > 0 else 0

        # ── Left: accuracy curves ──
        ax = axes[i, 0]
        ax.plot(layers, pooled, "o-", label="Pooled", color="#1976D2", linewidth=2)
        ax.plot(
            layers, per_pos, "s--", label="Per-position", color="#E65100", linewidth=2
        )
        ax.axhline(
            chance, color="gray", linestyle=":", alpha=0.6, label=f"Chance (1/{n_cls})"
        )
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(f"{rank}  ({n_cls} classes)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(bottom=0)

        # ── Right: gap ──
        ax = axes[i, 1]
        gap = pooled - per_pos
        colors = ["#4CAF50" if g >= 0 else "#F44336" for g in gap]
        ax.bar(layers, gap, color=colors, alpha=0.7, width=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Gap (pooled − per-position)")
        ax.set_title(f"{rank} — accuracy gap")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = OUTPUT_DIR / "probe_accuracy_by_layer.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out}")


def print_summary(results):
    """Print best layers per rank."""
    print(f"\n{'═' * 60}")
    print("SUMMARY: Best layers per rank")
    print(f"{'═' * 60}")

    best_layers = {}

    for rank, data in results.items():
        layers = data["layers"]
        pooled = data["pooled"]
        per_pos = data["per_position"]
        n_cls = data["n_classes"]
        chance = 1.0 / n_cls

        best_pool_idx = int(np.nanargmax(pooled))
        best_pp_idx = int(np.nanargmax(per_pos))

        print(f"\n{rank} ({n_cls} classes, chance={chance:.4f}):")
        print(
            f"  Best pooled:       layer {layers[best_pool_idx]:>2d}  "
            f"acc={pooled[best_pool_idx]:.4f}"
        )
        print(
            f"  Best per-position: layer {layers[best_pp_idx]:>2d}  "
            f"acc={per_pos[best_pp_idx]:.4f}"
        )
        gap_at_best_pool = pooled[best_pool_idx] - per_pos[best_pool_idx]
        print(f"  Gap at best pooled layer: {gap_at_best_pool:+.4f}")

        best_layers[rank] = {
            "best_pooled_layer": layers[best_pool_idx],
            "best_pooled_acc": pooled[best_pool_idx],
            "best_perpos_layer": layers[best_pp_idx],
            "best_perpos_acc": per_pos[best_pp_idx],
        }

    # Recommend layers to save for full run
    all_best = set()
    for rank, info in best_layers.items():
        all_best.add(info["best_pooled_layer"])
        all_best.add(info["best_perpos_layer"])

    # Also add neighbors of best layers (±1) since the optimum may shift
    # with more data
    neighbors = set()
    for l in all_best:
        if l > 0:
            neighbors.add(l - 1)
        if l < N_LAYERS - 1:
            neighbors.add(l + 1)
    recommended = sorted(all_best | neighbors)

    print(f"\n{'─' * 60}")
    print(f"Recommended layers to save for full run: {recommended}")
    print(
        f"({len(recommended)} layers — will reduce storage by "
        f"{(1 - len(recommended) / N_LAYERS) * 100:.0f}%)"
    )

    # Save summary
    summary = {"best_layers": best_layers, "recommended_layers": recommended}
    out = OUTPUT_DIR / "summary.pkl"
    with open(out, "wb") as f:
        pickle.dump(summary, f)
    print(f"Summary saved to {out}")


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    run_pilot()

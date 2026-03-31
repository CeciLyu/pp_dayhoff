"""
Two-stage GLM analysis of hierarchical probe per-position accuracy.

Stage 1: Binomial GLM on aggregated proportions with cluster-robust SE by OG.
         Covariates: phylum identity, position bin, context type, context size.
Stage 2: WLS on per-OG residuals from stage 1, regressed on OG-level features.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# Global settings
# ──────────────────────────────────────────────────────────────────────
# fpath = "/scratch/suyuelyu/deimm/pp_dayhoff/results/online_hierarchical_probe/lyr16_ntr1_nval10_lr1e-3_pat10_fact0.5_wd0_acc16_nmax1/per_pos_eval_hier/probe_taxon_per_pos_test_results_phylum_lyr16_checkpoint_best.pkl"
fpath = "/scratch/suyuelyu/deimm/pp_dayhoff/results/online_hierarchical_probe/lyr16_ntr1_nval10_lr1e-3_pat10_fact0.5_wd0_acc16_nmax24/per_pos_eval_hier/probe_taxon_per_pos_test_results_phylum_lyr16_checkpoint_best.pkl"

OG_FEATURES_CSV = "/scratch/suyuelyu/deimm/pp_dayhoff/script/analysis/oma_probe_og_features_full_with_manual9groups_test.csv"

N_BINS = 10

OG_FEATURES_STAGE2 = [
    "mean_seq_len",
    "n_unique_species",
    "species_per_phylum",
    "mean_pairwise_identity",
    "has_go_annotation",
    "nucleic_acid_binding",
    "catalytic",
    "regulatory_signaling",
    "nuclear",
    "cytoplasmic",
    "membrane_secretory",
    "endosymbiotic",
    "information_processing",
    "cellular_physiology",
]

IS_MULTI_SEQ = "nmax1" not in fpath

# Output directory: same as input pickle
OUTPUT_DIR = Path(fpath).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TAG = "multi" if IS_MULTI_SEQ else "single"


# ──────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────
def save_text(text: str, path: Path) -> None:
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved: {path}")


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Data reading
# ──────────────────────────────────────────────────────────────────────
def get_correct(pred_rank_per_pos, sample_ranks):
    return [pred == sample_ranks[-1] for pred in pred_rank_per_pos]


def read_data(fpath):
    with open(fpath, "rb") as f:
        phylum_per_pos = pickle.load(f)
    phylum_per_pos = pd.DataFrame(phylum_per_pos)
    # og collumn to int
    phylum_per_pos["og"] = pd.to_numeric(phylum_per_pos["og"], errors="coerce")

    phylum_per_pos["correct"] = phylum_per_pos.apply(
        lambda row: get_correct(row["pred_tgt_rank_id"], row["rank_ids"]), axis=1
    )
    phylum_per_pos["mean_accuracy"] = phylum_per_pos["correct"].apply(
        lambda x: sum(x) / len(x)
    )
    phylum_per_pos["n_src_proteins"] = phylum_per_pos["protein_names"].apply(
        lambda x: len(x) - 1 if x is not None else 0
    )

    if IS_MULTI_SEQ:
        phylum_per_pos["src_rank_id"] = phylum_per_pos["rank_ids"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else None
        )
        phylum_per_pos["tgt_rank_id"] = phylum_per_pos["rank_ids"].apply(
            lambda x: x[-1] if isinstance(x, (list, tuple)) and len(x) > 0 else None
        )
        phylum_per_pos["single_or_multi_rank"] = phylum_per_pos["rank_ids"].apply(
            lambda x: "single" if len(set(x)) == 1 else "multi"
        )

        # Keep only OGs that have at least one multi-rank sample
        og_w_multi_rank = phylum_per_pos.groupby("og")["single_or_multi_rank"].apply(
            lambda x: "multi" in x.values
        )
        phylum_per_pos_multi_rank = phylum_per_pos[
            phylum_per_pos["og"].isin(og_w_multi_rank[og_w_multi_rank].index)
        ]
        return phylum_per_pos, phylum_per_pos_multi_rank
    else:
        return phylum_per_pos, None


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Binomial GLM with position bins
# ──────────────────────────────────────────────────────────────────────
def aggregate_to_position_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-sample per-position results into position bins."""
    if IS_MULTI_SEQ:
        df = df[
            ["og", "tgt_rank_id", "single_or_multi_rank", "n_src_proteins", "correct"]
        ]
    else:
        df = df[["og", "rank_ids", "correct"]]

    valid = df["correct"].apply(
        lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
    )
    df = df[valid].copy()

    records = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Aggregating"):
        y = np.asarray(row.correct, dtype=np.int8)
        L = y.size

        if L == 1:
            rel = np.array([0.0], dtype=float)
        else:
            rel = np.arange(L, dtype=float) / (L - 1)

        b = np.floor(rel * N_BINS).astype(int)
        b = np.clip(b, 0, N_BINS - 1)

        trials = np.bincount(b, minlength=N_BINS)
        succ = np.bincount(b, weights=y, minlength=N_BINS)

        nz = trials > 0
        for bin_idx in np.where(nz)[0]:
            if IS_MULTI_SEQ:
                records.append(
                    (
                        row.og,
                        row.tgt_rank_id,
                        row.single_or_multi_rank,
                        row.n_src_proteins,
                        int(bin_idx),
                        int(succ[bin_idx]),
                        int(trials[bin_idx]),
                    )
                )
            else:
                records.append(
                    (
                        row.og,
                        row.rank_ids[-1],
                        int(bin_idx),
                        int(succ[bin_idx]),
                        int(trials[bin_idx]),
                    )  # type: ignore
                )

    if IS_MULTI_SEQ:
        agg = pd.DataFrame(
            records,
            columns=[
                "og",
                "tgt_rank_id",
                "single_or_multi_rank",
                "n_src_proteins",
                "pos_bin",
                "successes",
                "trials",
            ],
        )
        agg = agg.groupby(
            ["og", "tgt_rank_id", "single_or_multi_rank", "n_src_proteins", "pos_bin"],
            as_index=False,
        )[["successes", "trials"]].sum()
        agg["prop"] = agg["successes"] / agg["trials"]
        agg["n_src_proteins"] = pd.to_numeric(agg["n_src_proteins"], errors="coerce")
        agg = agg.dropna(
            subset=[
                "og",
                "tgt_rank_id",
                "single_or_multi_rank",
                "n_src_proteins",
                "prop",
                "trials",
            ]
        )
    else:
        agg = pd.DataFrame(
            records,
            columns=[
                "og",
                "rank_id",
                "pos_bin",
                "successes",
                "trials",
            ],
        )
        agg = agg.groupby(
            ["og", "rank_id", "pos_bin"],
            as_index=False,
        )[["successes", "trials"]].sum()
        agg["prop"] = agg["successes"] / agg["trials"]
        agg["rank_id"] = pd.to_numeric(agg["rank_id"], errors="coerce")
        agg = agg.dropna(subset=["og", "rank_id", "pos_bin", "prop", "trials"])

    return agg


def fit_stage1(agg: pd.DataFrame) -> dict:
    """Fit stage-1 GLMs and return results."""
    if IS_MULTI_SEQ:
        formula_no_pos = (
            "prop ~ C(tgt_rank_id) + C(single_or_multi_rank) * np.log(n_src_proteins)"
        )
    else:
        formula_no_pos = "prop ~ C(rank_id)"
    formula_with_pos = formula_no_pos + " + C(pos_bin)"

    # GLM without position bins
    glm_no_pos = smf.glm(
        formula=formula_no_pos,
        data=agg,
        family=sm.families.Binomial(),
        var_weights=agg["trials"],
    )
    res_no_pos = glm_no_pos.fit(
        cov_type="cluster",
        cov_kwds={"groups": agg["og"]},
    )

    # GLM with position bins
    glm_with_pos = smf.glm(
        formula=formula_with_pos,
        data=agg,
        family=sm.families.Binomial(),
        var_weights=agg["trials"],
    )
    res_with_pos = glm_with_pos.fit(
        cov_type="cluster",
        cov_kwds={"groups": agg["og"]},
    )

    # Joint Wald test for position bins
    param_names = list(res_with_pos.params.index)
    pos_terms = [name for name in param_names if name.startswith("C(pos_bin)")]
    wald_result = None
    if pos_terms:
        R = np.zeros((len(pos_terms), len(param_names)))
        for i, term in enumerate(pos_terms):
            R[i, param_names.index(term)] = 1.0
        wald_result = res_with_pos.wald_test(R)

    # Position-bin odds ratios
    or_df = pd.DataFrame(
        {
            "term": pos_terms,
            "beta": [res_with_pos.params[t] for t in pos_terms],
            "se": [res_with_pos.bse[t] for t in pos_terms],
        }
    )
    or_df["odds_ratio"] = np.exp(or_df["beta"])
    or_df["or_lo95"] = np.exp(or_df["beta"] - 1.96 * or_df["se"])
    or_df["or_hi95"] = np.exp(or_df["beta"] + 1.96 * or_df["se"])

    return {
        "res_no_pos": res_no_pos,
        "res_with_pos": res_with_pos,
        "wald_result": wald_result,
        "or_df": or_df,
        "formula_no_pos": formula_no_pos,
        "formula_with_pos": formula_with_pos,
    }


def save_stage1_results(stage1: dict, agg: pd.DataFrame) -> None:
    """Save stage 1 outputs to files."""
    report = []
    report.append(f"=== Stage 1 GLM Analysis ({TAG}) ===\n")
    report.append(f"Aggregated table shape: {agg.shape}\n")

    report.append(f"\n--- GLM WITHOUT position bins ---")
    report.append(f"Formula: {stage1['formula_no_pos']}")
    report.append(stage1["res_no_pos"].summary().as_text())

    report.append(f"\n--- GLM WITH position bins ---")
    report.append(f"Formula: {stage1['formula_with_pos']}")
    report.append(stage1["res_with_pos"].summary().as_text())

    if stage1["wald_result"] is not None:
        report.append(f"\n--- Joint Wald test for position bins ---")
        report.append(str(stage1["wald_result"]))

    report.append(f"\n--- Position-bin odds ratios (vs reference bin) ---")
    report.append(stage1["or_df"].to_string(index=False))

    save_text("\n".join(report), OUTPUT_DIR / f"stage1_results_{TAG}.txt")
    save_df(stage1["or_df"], OUTPUT_DIR / f"stage1_position_odds_ratios_{TAG}.csv")


# ──────────────────────────────────────────────────────────────────────
# Stage 2: OG-level residual analysis
# ──────────────────────────────────────────────────────────────────────
def compute_og_residuals(agg: pd.DataFrame, res_with_pos) -> pd.DataFrame:
    """
    Compute per-OG observed vs predicted accuracy, and residuals.
    Residuals = observed proportion - predicted proportion, aggregated per OG.
    """
    agg = agg.copy()
    agg["predicted_prop"] = res_with_pos.predict()

    # Weighted aggregation per OG
    og_obs = (
        agg.groupby("og")
        .apply(lambda g: np.average(g["prop"], weights=g["trials"]))
        .rename("observed_prop")
    )
    og_pred = (
        agg.groupby("og")
        .apply(lambda g: np.average(g["predicted_prop"], weights=g["trials"]))
        .rename("predicted_prop")
    )
    og_trials = agg.groupby("og")["trials"].sum().rename("total_trials")

    og_residuals = pd.concat([og_obs, og_pred, og_trials], axis=1).reset_index()
    og_residuals["residual"] = (
        og_residuals["observed_prop"] - og_residuals["predicted_prop"]
    )

    return og_residuals


def fit_stage2(og_residuals: pd.DataFrame, og_features: pd.DataFrame) -> dict:
    """Fit stage-2 WLS on OG-level residuals."""
    # Merge residuals with OG features
    merged = og_residuals.merge(og_features, on="og", how="left")

    # Check coverage
    n_total = len(merged)
    n_with_features = merged[OG_FEATURES_STAGE2[0]].notna().sum()
    print(f"Stage 2: {n_with_features}/{n_total} OGs have features")

    # Drop OGs without features (shouldn't happen if CSV covers all test OGs)
    merged = merged.dropna(subset=OG_FEATURES_STAGE2)
    print(f"Stage 2: {len(merged)} OGs after dropping NaN features")

    # Log-transform skewed continuous features
    merged["log_n_unique_species"] = np.log(merged["n_unique_species"].clip(lower=1))
    merged["log_mean_seq_len"] = np.log(merged["mean_seq_len"].clip(lower=1))

    # Build formula
    continuous_vars = [
        "log_n_unique_species",
        "species_per_phylum",
        "log_mean_seq_len",
        "mean_pairwise_identity",
    ]
    binary_vars = [
        f
        for f in OG_FEATURES_STAGE2
        if f
        not in [
            "n_unique_species",
            "mean_seq_len",
            "species_per_phylum",
            "mean_pairwise_identity",
        ]
    ]
    all_stage2_vars = continuous_vars + binary_vars
    formula = "residual ~ " + " + ".join(all_stage2_vars)

    # WLS weighted by total trials per OG
    wls = smf.wls(
        formula=formula,
        data=merged,
        weights=merged["total_trials"],
    )
    res_wls = wls.fit()

    # Also fit OLS for comparison
    ols = smf.ols(
        formula=formula,
        data=merged,
    )
    res_ols = ols.fit()

    # Cook's distance from OLS (diagnostic)
    influence = res_ols.get_influence()
    cooks_d, _ = influence.cooks_distance
    merged["cooks_d"] = cooks_d

    # Flag high-leverage OGs
    cooks_threshold = 4.0 / len(merged)
    high_leverage = merged[merged["cooks_d"] > cooks_threshold]

    return {
        "res_wls": res_wls,
        "res_ols": res_ols,
        "merged": merged,
        "formula": formula,
        "high_leverage": high_leverage,
        "cooks_threshold": cooks_threshold,
        "continuous_vars": continuous_vars,
        "binary_vars": binary_vars,
    }


def save_stage2_results(stage2: dict, og_residuals: pd.DataFrame) -> None:
    """Save stage 2 outputs to files."""
    report = []
    report.append(f"=== Stage 2 WLS Analysis ({TAG}) ===\n")
    report.append(f"Formula: {stage2['formula']}")
    report.append(f"N OGs: {len(stage2['merged'])}")
    report.append(f"Cook's distance threshold (4/n): {stage2['cooks_threshold']:.4f}")
    report.append(
        f"High-leverage OGs (Cook's d > threshold): {len(stage2['high_leverage'])}"
    )

    report.append(f"\n--- WLS Results (weighted by total trials) ---")
    report.append(stage2["res_wls"].summary().as_text())

    report.append(f"\n--- OLS Results (unweighted, for comparison) ---")
    report.append(stage2["res_ols"].summary().as_text())

    # Residual diagnostics
    residuals = stage2["res_wls"].resid
    report.append(f"\n--- Stage 2 residual diagnostics ---")
    report.append(f"Residual mean: {residuals.mean():.6f}")
    report.append(f"Residual std: {residuals.std():.6f}")
    report.append(f"Shapiro-Wilk p-value: {stats.shapiro(residuals[:500])[1]:.4f}")

    # High-leverage OGs
    if len(stage2["high_leverage"]) > 0:
        report.append(f"\n--- Top 20 high-leverage OGs ---")
        top_leverage = stage2["high_leverage"].nlargest(20, "cooks_d")
        cols_to_show = [
            "og",
            "observed_prop",
            "predicted_prop",
            "residual",
            "total_trials",
            "cooks_d",
        ]
        report.append(top_leverage[cols_to_show].to_string(index=False))

    save_text("\n".join(report), OUTPUT_DIR / f"stage2_results_{TAG}.txt")
    save_df(og_residuals, OUTPUT_DIR / f"stage2_og_residuals_{TAG}.csv")
    save_df(stage2["merged"], OUTPUT_DIR / f"stage2_merged_{TAG}.csv")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Analysis mode: {'multi-sequence' if IS_MULTI_SEQ else 'single-sequence'}")
    print(f"Input: {fpath}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Read data
    phylum_per_pos, phylum_per_pos_multi_rank = read_data(fpath)

    # Stage 1
    print("\n" + "=" * 60)
    print("STAGE 1: Binomial GLM")
    print("=" * 60)

    # if IS_MULTI_SEQ:
    #     agg = aggregate_to_position_bins(phylum_per_pos_multi_rank)
    # else:
    #     agg = aggregate_to_position_bins(phylum_per_pos)
    agg = aggregate_to_position_bins(phylum_per_pos)

    stage1 = fit_stage1(agg)
    save_stage1_results(stage1, agg)

    # Print key stage 1 results to stdout as well
    print(f"\nStage 1 summary:")
    print(stage1["res_with_pos"].summary())

    # Stage 2
    print("\n" + "=" * 60)
    print("STAGE 2: OG-level residual analysis")
    print("=" * 60)

    og_residuals = compute_og_residuals(agg, stage1["res_with_pos"])
    print(f"OG residuals computed: {len(og_residuals)} OGs")
    print(
        f"Residual range: [{og_residuals['residual'].min():.4f}, "
        f"{og_residuals['residual'].max():.4f}]"
    )

    og_features = pd.read_csv(OG_FEATURES_CSV)
    # convert og collumn to int
    og_features["og"] = pd.to_numeric(og_features["og"], errors="coerce")
    print(
        f"OG features loaded: {len(og_features)} OGs, "
        f"columns: {og_features.columns.tolist()}"
    )

    stage2 = fit_stage2(og_residuals, og_features)
    save_stage2_results(stage2, og_residuals)

    # Print key stage 2 results to stdout
    print(f"\nStage 2 WLS summary:")
    print(stage2["res_wls"].summary())
    print(f"\nHigh-leverage OGs: {len(stage2['high_leverage'])}")

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

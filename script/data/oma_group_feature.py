import argparse
import itertools
import os
import random
import pickle
import re
from glob import glob
import numpy as np
import pandas as pd
import parasail
from goatools.obo_parser import GODag
from goatools.mapslim import mapslim
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


GAP_OPEN = 11
GAP_EXTEND = 1
MATRIX = parasail.blosum62


RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

DEFAULT_TAXONOMY_MAPPING = "/scratch/suyuelyu/deimm/data/oma/taxid_to_std_ranks.pkl"
DEFAULT_LINCLUST_50_FILE = (
    "/scratch/suyuelyu/deimm/data/oma/oma-seqs-linclust-50pident_cluster.tsv"
)
N_CHUNKS = 5


def _is_parquet_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".parquet", ".pq"}


def _chunk_path(path: str, idx: int) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_chunk{idx}{ext}"


def _chunk_paths(path: str) -> list[str]:
    base, ext = os.path.splitext(path)
    paths = glob(f"{base}_chunk*{ext}")

    def _chunk_idx(p: str) -> int:
        m = re.search(r"_chunk(\d+)$", os.path.splitext(p)[0])
        return int(m.group(1)) if m else 10**9

    return sorted(paths, key=_chunk_idx)


def _read_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if _is_parquet_path(path) else pd.read_csv(path)


def _write_df(df: pd.DataFrame, path: str, index: bool = False):
    if _is_parquet_path(path):
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)


def output_exists(path: str) -> bool:
    return os.path.exists(path) or len(_chunk_paths(path)) > 0


def load_output_df(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return _read_df(path)
    paths = _chunk_paths(path)
    if not paths:
        raise FileNotFoundError(path)
    return pd.concat([_read_df(p) for p in paths], ignore_index=True)


def save_output_df(df: pd.DataFrame, path: str, index: bool = False):
    if N_CHUNKS > 1:
        chunk_size = max(1, (len(df) + N_CHUNKS - 1) // N_CHUNKS)
        for i in range(N_CHUNKS):
            chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk_df) == 0:
                continue
            _write_df(chunk_df, _chunk_path(path, i + 1), index=index)
        return
    _write_df(df, path, index=index)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build OMA OG features with optional step-wise execution."
    )
    parser.add_argument(
        "--input-parquet",
        default="/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet",
        help="Input grouped test parquet.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        help="Steps to run: 1 2 3 merge (or all). Accepts space/comma-separated values.",
    )

    # Step 1
    parser.add_argument("--step1-out", default="oma_probe_og_features_test.csv")

    # Step 2
    parser.add_argument(
        "--step2-raw-out", default="oma_probe_og_pairwise_identity_raw.pkl"
    )
    parser.add_argument(
        "--step2-out", default="oma_probe_og_features_with_identity_test.csv"
    )
    parser.add_argument("--max-proteins", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))

    # Step 3
    parser.add_argument(
        "--oma-go-path", default="/scratch/suyuelyu/deimm/data/oma/oma-go.txt"
    )
    parser.add_argument(
        "--go-obo", default="/scratch/suyuelyu/deimm/data/oma/go-basic.obo"
    )
    parser.add_argument(
        "--slim-obo", default="/scratch/suyuelyu/deimm/data/oma/goslim_generic.obo"
    )
    parser.add_argument("--og-slim-out", default="og_go_slim_terms.csv")
    parser.add_argument("--og-func-out", default="og_functional_categories_test.csv")

    # Merge
    parser.add_argument("--full-out", default="oma_probe_og_features_full_test.csv")
    return parser.parse_args()


# Loaded input parquet: /scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq.parquet (14104344 rows)
# Columns: ['protein', 'taxid', 'lineage', 'og', 'seq']
# Input parquet appears to be on single sequence level. Aggregating to OG level.
# Warning: Found 788254 sequences longer than 1024 aa. These will be filtered out before OG aggregation.
# After filtering long sequences: 13316090 rows remain.
# Aggregated to OG level: 678795 rows
# Warning: Found 5484 OGs with fewer than 2 proteins. These will be dropped since pairwise identity cannot be computed.
# After dropping singletons: 673311 OGs remain.
# Index(['og', 'seq', 'protein', 'taxid', 'lineage', 'n_proteins', 'domain',
#        'phylum', 'class', 'order', 'family', 'genus', 'species', 'cluster_id'],
#       dtype='object')


def normalize_steps(raw_steps):
    parts = []
    for s in raw_steps:
        parts.extend(x.strip().lower() for x in s.split(",") if x.strip())
    steps = set(parts)
    if "all" in steps:
        return {"1", "2", "3", "merge"}
    invalid = steps - {"1", "2", "3", "merge"}
    if invalid:
        raise ValueError(f"Invalid --steps values: {sorted(invalid)}")
    return steps


def load_rank_mapping(mapping_file: str) -> dict[str, dict[int, int]]:
    with open(mapping_file, "rb") as f:
        taxid_to_std_ranks = pickle.load(f)
    rank_mapping: dict[str, dict[int, int]] = {}
    for rank in RANKS:
        rank_mapping[rank] = {}
    for species_tid, rank_dict in taxid_to_std_ranks.items():
        for rank, value in rank_dict.items():
            rank_mapping[rank][int(species_tid)] = int(value)
    return rank_mapping


def load_linclust_50_clusters(cluster_file: str) -> dict[str, int]:
    linclust_df = pd.read_csv(
        cluster_file, sep="\t", header=None, names=["rep_id", "seq_id"]
    )
    cluster_sizes = linclust_df["rep_id"].value_counts()
    sorted_reps = cluster_sizes.index.tolist()
    rep_to_cluster_id = {rep: f"cluster_{i}" for i, rep in enumerate(sorted_reps)}
    linclust_df["cluster_id"] = linclust_df["rep_id"].map(rep_to_cluster_id)
    seq_to_cluster_id = dict(zip(linclust_df["seq_id"], linclust_df["cluster_id"]))
    return seq_to_cluster_id


def process_input_parquet(input_parquet):
    df = pd.read_parquet(input_parquet)
    print(f"Loaded input parquet: {input_parquet} ({len(df)} rows)")
    print(f"Columns: {df.columns.tolist()}")

    # check if input parquet is on OG level or single seq level
    # if seq colum first element is str: seq level, if list: OG level
    if len(df) == 0:
        raise ValueError("Input parquet is empty.")
    # expect col: og, seq (list), protein (list), taxid (list), lineage (list)
    required_cols = {"og", "seq", "protein", "taxid", "lineage"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns for OG level: {required_cols}. Found: {set(df.columns)}"
        )

    IS_SEQ_LEVEL = isinstance(df.iloc[0]["seq"], str)
    if IS_SEQ_LEVEL:
        print(
            "Input parquet appears to be on single sequence level. Aggregating to OG level."
        )
        # filter out any seq longer than 1024 aa before aggregation to save memory and avoid issues in pairwise identity step
        df["seq_len"] = df["seq"].apply(len)
        n_long = (df["seq_len"] > 1024).sum()
        if n_long > 0:
            print(
                f"Warning: Found {n_long} sequences longer than 1024 aa. These will be filtered out before OG aggregation."
            )
            df = df[df["seq_len"] <= 1024]
            print(f"After filtering long sequences: {len(df)} rows remain.")
            # drop seq_len column after filtering
            df = df.drop(columns=["seq_len"])

        # group by OG and aggregate sequences and other info into lists
        df = (
            df.groupby("og")
            .agg(
                {
                    "seq": list,
                    "protein": list,
                    "taxid": list,
                    "lineage": list,
                }
            )
            .reset_index()
        )
        print(f"Aggregated to OG level: {len(df)} rows")
    else:
        print("Input parquet appears to be on OG level.")

    # assert no seq longer than 1024 aa
    max_len = df["seq"].apply(lambda seqs: max(len(s) for s in seqs)).max()
    if max_len > 1024:
        raise ValueError(f"Found sequence longer than 1024 aa. Max length: {max_len}")

    # drop any OG with fewer than 2 proteins since we can't compute pairwise identity
    df["n_proteins"] = df["protein"].apply(len)
    n_singletons = (df["n_proteins"] < 2).sum()
    if n_singletons > 0:
        print(
            f"Warning: Found {n_singletons} OGs with fewer than 2 proteins. These will be dropped since pairwise identity cannot be computed."
        )
        df = df[df["n_proteins"] >= 2]
        print(f"After dropping singletons: {len(df)} OGs remain.")

    # if no column domain	phylum	class	order	family	genus	species , add them using the lineage mapping file
    if not all(col in df.columns for col in RANKS):
        rank_mappings = load_rank_mapping(DEFAULT_TAXONOMY_MAPPING)
        for rank in RANKS:
            if rank not in df.columns:
                df[rank] = df["taxid"].apply(
                    lambda taxids: [
                        rank_mappings[rank].get(tid, np.nan) for tid in taxids
                    ]
                )

    # if missing colum: cluster_id
    if "cluster_id" not in df.columns:
        seq_to_cluster_id = load_linclust_50_clusters(DEFAULT_LINCLUST_50_FILE)
        df["cluster_id"] = df["protein"].apply(
            lambda pids: list({seq_to_cluster_id.get(pid, "singleton") for pid in pids})
        )

    if IS_SEQ_LEVEL:
        # save aggregated OG-level output using shared output helper
        og_level_path = f"{os.path.splitext(input_parquet)[0]}_probe_og.parquet"
        save_output_df(df, og_level_path, index=False)
    return df


def pairwise_identity(s1: str, s2: str) -> float:
    """Compute fractional identity from global alignment."""
    result = parasail.nw_stats(s1, s2, GAP_OPEN, GAP_EXTEND, MATRIX)
    # result.matches = number of identical positions
    # result.length = alignment length (including gaps)
    return result.matches / result.length if result.length > 0 else 0.0


def _compute_og_identity(task):
    og, seqs, names, max_proteins, seed = task

    random.seed(seed)
    matrix = parasail.blosum62
    gap_open, gap_extend = 11, 1

    n = len(seqs)
    if n < 2:
        return og, [], np.nan

    if n > max_proteins:
        idxs = random.sample(range(n), max_proteins)
        seqs = [seqs[i] for i in idxs]
        names = [names[i] for i in idxs]
        n = max_proteins

    pairs_results = []
    identities = []
    for i, j in itertools.combinations(range(n), 2):
        r = parasail.nw_stats(seqs[i], seqs[j], gap_open, gap_extend, matrix)
        pid = (r.matches / r.length) if r.length > 0 else 0.0
        pairs_results.append((names[i], names[j], pid))
        identities.append(pid)

    return og, pairs_results, float(np.mean(identities)) if identities else np.nan


def main():
    args = parse_args()
    selected = normalize_steps(args.steps)

    run_step1 = "1" in selected
    run_step2 = "2" in selected
    run_step3 = "3" in selected
    run_merge = "merge" in selected

    df = process_input_parquet(args.input_parquet)
    print(df.columns)

    og_features = None
    summary_df = None
    og_slim_df = None
    og_func_df = None
    og_slim_terms = defaultdict(set)

    # ===============================================================================
    # STEP 1
    # ===============================================================================
    if run_step1:
        print("\n[STEP 1] Compute OG-level trivial features")
        if not output_exists(args.step1_out):
            records = []
            col_idx = {c: i for i, c in enumerate(df.columns)}

            for row in tqdm(
                df.itertuples(index=False, name=None),
                total=len(df),
                desc="STEP1: OG features",
                unit="og",
            ):
                rec = {
                    "og": row[col_idx["og"]],
                    "n_proteins": len(row[col_idx["seq"]]),
                    "mean_seq_len": np.mean([len(s) for s in row[col_idx["seq"]]]),
                    "cluster_id": row[col_idx["cluster_id"]],
                }
                for rank in RANKS:
                    vals = row[col_idx[rank]]
                    if rank == "species":
                        rec[f"n_unique_{rank}"] = len(
                            set(int(t) for t in row[col_idx["taxid"]])
                        )
                    else:
                        clean = [int(v) for v in vals if not np.isnan(v)]
                        rec[f"n_unique_{rank}"] = len(set(clean))
                records.append(rec)

            og_features = pd.DataFrame(records)
            save_output_df(og_features, args.step1_out, index=False)
            print(f"[STEP 1] Saved: {args.step1_out} ({len(og_features)} rows)")
        else:
            og_features = load_output_df(args.step1_out)
            print(
                f"[STEP 1] Loaded existing: {args.step1_out} ({len(og_features)} rows)"
            )

    # ===============================================================================
    # STEP 2
    # ===============================================================================
    if run_step2:
        if og_features is None:
            if output_exists(args.step1_out):
                og_features = load_output_df(args.step1_out)
            else:
                raise FileNotFoundError(
                    f"Step 2 needs Step 1 features. Missing: {args.step1_out}. "
                    "Run with --steps 1 2 or generate step1 output first."
                )

        if output_exists(args.step2_out):
            summary_df = load_output_df(args.step2_out)
            print(
                f"[STEP 2] Loaded existing: {args.step2_out} ({len(summary_df)} rows)"
            )
        else:
            if os.path.exists(args.step2_raw_out):
                with open(args.step2_raw_out, "rb") as f:
                    all_og_pident = pickle.load(f)
                print(
                    f"[STEP 2] Loaded cached raw pairwise identities for {len(all_og_pident)} OGs"
                )

                # Rebuild means from cached raw data
                og_mean_pident = {}
                for og, pairs in tqdm(
                    all_og_pident.items(),
                    desc="STEP2: mean identity from cache",
                    unit="og",
                ):
                    if len(pairs) == 0:
                        og_mean_pident[og] = np.nan
                    else:
                        og_mean_pident[og] = float(np.mean([p[2] for p in pairs]))
            else:
                MAX_PROTEINS = 200
                SEED = 42

                tasks = [
                    (row.og, list(row.seq), list(row.protein), MAX_PROTEINS, SEED)
                    for row in df.itertuples(index=False)
                ]
                all_og_pident, og_mean_pident = {}, {}
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    for og, pairs_results, mean_pid in tqdm(
                        ex.map(_compute_og_identity, tasks, chunksize=8),
                        total=len(tasks),
                        desc="STEP2: pairwise identity (mp)",
                        unit="og",
                    ):
                        all_og_pident[og] = pairs_results
                        og_mean_pident[og] = mean_pid

                with open(args.step2_raw_out, "wb") as f:
                    pickle.dump(all_og_pident, f)
                print(f"[STEP 2] Saved cache: {args.step2_raw_out}")

            pident_df = pd.DataFrame(
                [
                    {"og": og, "mean_pairwise_identity": val}
                    for og, val in og_mean_pident.items()
                ]
            )
            pident_df["og"] = pident_df["og"].astype(int)
            summary_df = og_features.merge(pident_df, on="og", how="left")
            save_output_df(summary_df, args.step2_out, index=False)
            print(f"[STEP 2] Saved: {args.step2_out} ({len(summary_df)} rows)")

    # ===============================================================================
    # STEP 3
    # ===============================================================================
    if run_step3:
        if output_exists(args.og_slim_out):
            og_slim_df = load_output_df(args.og_slim_out)
            print(
                f"[STEP 3] Loaded existing: {args.og_slim_out} ({len(og_slim_df)} rows)"
            )
            # Rebuild mapping so functional category step works even when loading cached slim file
            for row in og_slim_df.itertuples(index=False):
                og_slim_terms[int(row.og)].add(row.go_slim)
        else:
            # ── Paths ──
            OMA_GO_PATH = args.oma_go_path

            # ── Step 3.1: Collect all protein IDs in OGs ──
            protein_to_og = {}
            for row in df.itertuples(index=False):
                for pid in row.protein:
                    protein_to_og[pid] = row.og

            proteins = set(protein_to_og.keys())
            print(f"Total proteins in input: {len(proteins)}")

            # ── Step 3.2: Parse oma-go.txt, filtering to proteins only ──
            go_records = []
            print(f"[STEP 3] Parsing GO file: {OMA_GO_PATH}")
            with open(OMA_GO_PATH, "r") as f:
                for line in tqdm(f, desc="STEP3: read oma-go.txt", unit="line"):
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    protein_id, go_term, evidence = parts[0], parts[1], parts[2]
                    if protein_id in proteins:
                        go_records.append(
                            {
                                "protein_id": protein_id,
                                "og": protein_to_og[protein_id],
                                "go_term": go_term,
                                "evidence": evidence,
                            }
                        )

            go_df = pd.DataFrame(go_records)
            print(f"GO annotations for input proteins: {len(go_df)}")
            print(f"OGs with any GO annotation: {go_df['og'].nunique()}")
            print(f"Total OGs in input: {df['og'].nunique()}")

            # ── Step 3.3: Map GO terms to GO slim using goatools ──
            OBO_DIR = "/scratch/suyuelyu/deimm/data/oma/"
            GO_OBO = os.path.join(OBO_DIR, "go-basic.obo")
            SLIM_OBO = os.path.join(OBO_DIR, "goslim_generic.obo")

            godag = GODag(GO_OBO)
            slim_dag = GODag(SLIM_OBO)

            go_to_slim_cache = {}

            def get_slim_terms(go_term):
                if go_term in go_to_slim_cache:
                    return go_to_slim_cache[go_term]
                if go_term not in godag:
                    go_to_slim_cache[go_term] = set()
                    return set()
                direct, all_anc = mapslim(go_term, godag, slim_dag)
                go_to_slim_cache[go_term] = direct
                return direct

            # ── Step 3.4: Aggregate GO slim terms per OG ──
            og_slim_terms = defaultdict(set)
            for _, row in tqdm(
                go_df.iterrows(),
                total=len(go_df),
                desc="STEP3: map GO->slim",
                unit="ann",
            ):
                slims = get_slim_terms(row["go_term"])
                og_slim_terms[row["og"]].update(slims)

            # ── Step 3.5: Create coarse functional categories ──
            slim_counter = defaultdict(int)
            for og, slims in og_slim_terms.items():
                for s in slims:
                    slim_counter[s] += 1

            slim_summary = pd.DataFrame(
                [
                    {
                        "go_slim": s,
                        "n_ogs": n,
                        "name": godag[s].name if s in godag else "?",
                    }
                    for s, n in slim_counter.items()
                ]
            ).sort_values("n_ogs", ascending=False)

            print("\nTop 30 GO slim terms across OGs:")
            print(slim_summary.head(30).to_string(index=False))

            og_slim_records = []
            for og, slims in og_slim_terms.items():
                for s in slims:
                    og_slim_records.append(
                        {
                            "og": og,
                            "go_slim": s,
                            "go_slim_name": godag[s].name if s in godag else "?",
                        }
                    )

            og_slim_df = pd.DataFrame(og_slim_records)
            save_output_df(og_slim_df, args.og_slim_out, index=False)
            print(
                f"\nTotal unique GO slim terms in OGs: {og_slim_df['go_slim'].nunique()}"
            )
            print(f"[STEP 3] Saved: {args.og_slim_out} ({len(og_slim_df)} rows)")

        if output_exists(args.og_func_out):
            og_func_df = load_output_df(args.og_func_out)
            og_func_df["og"] = og_func_df["og"].astype(int)
            print(
                f"[STEP 3] Loaded existing: {args.og_func_out} ({len(og_func_df)} rows)"
            )
        else:
            COARSE_CATEGORIES = {
                "catalytic": {"GO:0003824"},
                "binding": {"GO:0005488"},
                "membrane": {"GO:0016020"},
                "transporter": {"GO:0005215"},
                "signaling": {"GO:0023052"},
                "regulation": {"GO:0065007"},
                "metabolic": {"GO:0008152"},
                "cellular_component_org": {"GO:0016043"},
            }

            og_func_records = []
            all_ogs = set(df["og"])
            for og in tqdm(
                all_ogs,
                total=len(all_ogs),
                desc="STEP3: functional categories",
                unit="og",
            ):
                slims = og_slim_terms.get(og, set())
                rec = {"og": og, "has_go_annotation": len(slims) > 0}
                for cat_name, cat_terms in COARSE_CATEGORIES.items():
                    rec[cat_name] = int(bool(slims & cat_terms))
                og_func_records.append(rec)

            og_func_df = pd.DataFrame(og_func_records)
            og_func_df["og"] = og_func_df["og"].astype(int)
            save_output_df(og_func_df, args.og_func_out, index=False)
            print(f"[STEP 3] Saved: {args.og_func_out} ({len(og_func_df)} rows)")
            print(
                f"\nGO annotation coverage: {og_func_df['has_go_annotation'].mean():.1%} of OGs"
            )
            print("\nCategory prevalence:")
            for cat in COARSE_CATEGORIES:
                print(f"  {cat}: {og_func_df[cat].mean():.1%}")

    # ===============================================================================
    # MERGE
    # ===============================================================================
    if run_merge:
        if summary_df is None:
            if output_exists(args.step2_out):
                summary_df = load_output_df(args.step2_out)
            else:
                raise FileNotFoundError(
                    f"Merge needs Step 2 output. Missing: {args.step2_out}. "
                    "Run --steps 2 merge (or all) first."
                )
        if og_func_df is None:
            if output_exists(args.og_func_out):
                og_func_df = load_output_df(args.og_func_out)
                og_func_df["og"] = og_func_df["og"].astype(int)
            else:
                raise FileNotFoundError(
                    f"Merge needs Step 3 functional output. Missing: {args.og_func_out}."
                )
        if og_slim_df is None:
            if output_exists(args.og_slim_out):
                og_slim_df = load_output_df(args.og_slim_out)
            else:
                raise FileNotFoundError(
                    f"Merge needs Step 3 slim output. Missing: {args.og_slim_out}."
                )

        if not output_exists(args.full_out):
            full_df = summary_df.merge(og_func_df, on="og", how="left")
            slim_agg = (
                og_slim_df.groupby("og")["go_slim"]
                .apply(lambda x: ";".join(x))
                .reset_index()
            )
            slim_agg["og"] = slim_agg["og"].astype(int)
            full_df = full_df.merge(slim_agg, on="og", how="left")
            save_output_df(full_df, args.full_out, index=False)
            print(f"[MERGE] Saved: {args.full_out} ({len(full_df)} rows)")
        else:
            full_df = load_output_df(args.full_out)
            print(f"[MERGE] Loaded existing: {args.full_out} ({len(full_df)} rows)")


if __name__ == "__main__":
    main()

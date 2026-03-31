import pandas as pd
from goatools.obo_parser import GODag
import os
import glob
from oma_group_feature import save_output_df

# --step1-out /scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq_probe_og_trivial_features.parquet \
# --og-slim-out /scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq_probe_og_goslim.parquet \
# --og-func-out /scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq_probe_og_func.parquet
oma_trivial_features_path = "/scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq_probe_og_trivial_features_chunk*.parquet"
oma_og_slim_out_path = (
    "/scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq_probe_og_goslim_chunk*.parquet"
)
oma_og_func_out_path = (
    "/scratch/suyuelyu/deimm/data/oma/metadata_all_w_seq_probe_og_func_chunk*.parquet"
)
OBO_DIR = "/scratch/suyuelyu/deimm/data/oma/"
GO_OBO = os.path.join(OBO_DIR, "go-basic.obo")
SLIM_OBO = os.path.join(OBO_DIR, "goslim_generic.obo")

godag = GODag(GO_OBO)

MIN_COUNT = 5000
MAX_COUNT = 40000
CORR_THRESHOLD = 0.5

N_CHUNKS = 5


def get_parent_counts(non_grouped_og_slim_df, go_level):
    # slim_counts is counts of each slim in og_slim_df, which is the number of ogs annotated with that slim divided by total number of ogs
    slim_counts = non_grouped_og_slim_df["go_slim"].value_counts()

    # get parent at go_level for each slim
    slim_to_parents = {}
    for slim in slim_counts.index:
        if slim in godag:
            self_leve = godag[slim].level
            if self_leve == go_level:
                # check if the slim itself is at the go_level, if so, add it to the parents list
                slim_to_parents[slim] = [slim]
            elif self_leve > go_level:
                # if the slim is below the go_level, use parents at the go_level as the parents
                parents = godag[slim].get_all_parents()
                slim_to_parents[slim] = [
                    p for p in parents if godag[p].level == go_level
                ]
            else:
                # ignore
                continue
        else:
            continue

    # get the counts of each parent slim by summing the counts of its child slims
    parent_slim_counts = {}
    for slim, parents in slim_to_parents.items():
        for parent in parents:
            if parent not in parent_slim_counts:
                parent_slim_counts[parent] = 0
            parent_slim_counts[parent] += slim_counts[slim]

    return parent_slim_counts, slim_to_parents


def load_merge_data(
    oma_trivial_features_path, oma_og_slim_out_path, oma_og_func_out_path, go_levels
):
    # glob the files matching the patterns and read them into dataframes
    trivial_files = glob.glob(oma_trivial_features_path)
    og_slim_files = glob.glob(oma_og_slim_out_path)
    og_func_files = glob.glob(oma_og_func_out_path)
    trivial_df = pd.concat(
        [pd.read_parquet(f) for f in trivial_files], ignore_index=True
    )
    og_slim_df = pd.concat(
        [pd.read_parquet(f) for f in og_slim_files], ignore_index=True
    )
    go_parent_count_at_level = {
        level: get_parent_counts(og_slim_df, level) for level in go_levels
    }
    # og_slim_df need to be grouby og and aggregate other columns to list
    og_slim_df_grouped = og_slim_df.groupby("og").agg(lambda x: list(x)).reset_index()
    og_func_df = pd.concat(
        [pd.read_parquet(f) for f in og_func_files], ignore_index=True
    )
    # print original shapes of the dataframes
    print(f"Trivial features dataframe shape: {trivial_df.shape}")
    print(f"OG GO slim dataframe shape: {og_slim_df_grouped.shape}")
    print(f"OG functional annotations dataframe shape: {og_func_df.shape}")

    # Merge the dataframes on 'og'
    # make sure og has compatible types in all dataframes
    trivial_df["og"] = trivial_df["og"].astype(int)
    og_slim_df_grouped["og"] = og_slim_df_grouped["og"].astype(int)
    og_func_df["og"] = og_func_df["og"].astype(int)

    # Merge the dataframes on 'og', keep any og that appears in any of the dataframes
    merged_df = pd.merge(trivial_df, og_slim_df_grouped, on="og", how="outer")
    merged_df["go_slim"] = merged_df["go_slim"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged_df["go_slim_name"] = merged_df["go_slim_name"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    merged_df = pd.merge(merged_df, og_func_df, on="og", how="outer")
    print(f"Merged dataframe shape: {merged_df.shape}")

    return merged_df, go_parent_count_at_level, og_slim_df


def filter_slims_by_count_cooccurence(
    go_parent_count_at_level,
    og_slim_df,
    min_count=MIN_COUNT,
    max_count=MAX_COUNT,
    corr_threshold=CORR_THRESHOLD,
):
    """
    Filter GO terms using hierarchical count thresholds and co-occurrence pruning.

    This function traverses GO parent terms from higher to lower levels using
    `go_parent_count_at_level` and applies:

    - `count < min_count`: exclude the parent and all descendants.
    - `count > max_count`: skip the parent (defer to more specific children at lower levels).
    - otherwise: keep the parent as a candidate manual GO term.

    After candidate selection, it computes expanded counts per retained GO term
    (term + descendants, based on `go_slim_counts`) and removes highly co-occurring
    terms using a correlation threshold on OG-by-GO presence (`og_slim_df`).

    Args:
        go_parent_count_at_level (dict): GO term level -> (parent_counts, slim_to_parents).
            parent_counts: dict of parent GO term at this level -> sum counts of it and its children.
            slim_to_parents: dict of slim GO term -> list of parent GO terms at that level.
        og_slim_df (pd.DataFrame): Long-format table with at least ['og', 'go_slim'].
        min_count (int, optional): Minimum count required to keep a GO term.
        max_count (int, optional): Maximum count before deferring to children.
        corr_threshold (float, optional): Correlation cutoff for co-occurrence filtering.

    Returns:
        tuple:
            - manual_go_counts (dict): Aggregated counts for `manual_gos`.
            - go_slim_to_oma_group (dict): Mapping of GO slim terms to sets of OMA groups.
            - og_by_go (pd.DataFrame): Binary matrix of OGs by retained manual GO terms after co-occurrence pruning.
    """
    # start from the highest level and go down to the lowest level
    manual_gos = []
    excluded_gos = []
    for level in sorted(go_parent_count_at_level.keys(), reverse=False):
        print(f"level: {level}")
        parent_counts, _ = go_parent_count_at_level[level]
        for parent, count in parent_counts.items():
            # if parent == "GO:0043226" or parent == "GO:0043228":
            #     # debug print both list
            #     print(f"Parent: {parent}, Count: {count}")
            #     print(f'manual_gos: {manual_gos}')
            #     print(f'excluded_gos: {excluded_gos}')
            if parent in excluded_gos:
                continue
            if count < min_count:
                # remove this parent and its children
                excluded_gos.append(parent)
                excluded_gos.extend(godag[parent].get_all_children())
            elif count > max_count:
                continue
            else:
                manual_gos.append(parent)
                # exclude its children from being selected as manual gos in the future
                excluded_gos.extend(godag[parent].get_all_children())

    # get a counter for manual_gos: sum of all childrent in go_slim_counts
    go_slim_counts = og_slim_df["go_slim"].value_counts()
    go_slim_to_oma_group = og_slim_df.groupby("go_slim")["og"].apply(set).to_dict()
    manual_go_counts = {}
    for go in manual_gos:
        manual_go_counts[go] = 0
        if go in go_slim_counts:
            manual_go_counts[go] += go_slim_counts[go]
        for child in godag[go].get_all_children():
            if child in go_slim_counts:
                manual_go_counts[go] += go_slim_counts[child]
    print(f"Selected {len(manual_gos)} manual GO terms after count filtering.")

    # filter out any coocurring GOs within manual_go list (correlation > 0.5 in og_slim_df)
    og_slim_df["go_slim_w_all_parents"] = og_slim_df["go_slim"].apply(
        lambda x: godag[x].get_all_parents().union(set([x])) if x in godag else set([x])
    )
    og_slim_df = og_slim_df.drop("go_slim", axis=1)
    og_slim_df = og_slim_df.explode("go_slim_w_all_parents")
    og_slim_df_manual = og_slim_df[og_slim_df["go_slim_w_all_parents"].isin(manual_gos)]
    assert og_slim_df_manual["go_slim_w_all_parents"].nunique() == len(manual_gos)
    og_slim_df_manual["presence"] = 1
    og_by_go = og_slim_df_manual.pivot_table(
        index="og", columns="go_slim_w_all_parents", values="presence", fill_value=0
    )  # row: og, column: go, value: 1 if the og is annotated with this go or its children, otherwise 0
    cooccurrence_matrix_corr = og_by_go.corr()
    print(
        f"Constructed correlation matrix with shape: {cooccurrence_matrix_corr.shape}"
    )

    to_remove = set()
    for i in range(len(cooccurrence_matrix_corr)):
        for j in range(i + 1, len(cooccurrence_matrix_corr)):
            if (
                cooccurrence_matrix_corr.index[i] in to_remove
                or cooccurrence_matrix_corr.columns[j] in to_remove
            ):
                continue
            if cooccurrence_matrix_corr.iloc[i, j] > corr_threshold:  # type: ignore
                go_i = cooccurrence_matrix_corr.index[i]
                go_j = cooccurrence_matrix_corr.columns[j]
                # remove the lower-count term
                if manual_go_counts[go_i] >= manual_go_counts[go_j]:
                    to_remove.add(go_j)
                else:
                    to_remove.add(go_i)

    manual_gos = [g for g in manual_gos if g not in to_remove]
    manual_go_counts = {go: manual_go_counts[go] for go in manual_gos}

    # filter og_by_go to only keep columns in manual_gos
    og_by_go = og_by_go[[go for go in og_by_go.columns if go in manual_gos]]
    # drop columns with all 0s
    og_by_go = og_by_go.loc[:, (og_by_go != 0).any(axis=0)]
    print(
        f"Final selected {len(manual_gos)} manual GO terms after co-occurrence pruning."
    )

    return (manual_go_counts, go_slim_to_oma_group, og_by_go)


def main():
    oma_og_features_df, go_parent_count_at_level, og_slim_df = load_merge_data(
        oma_trivial_features_path,
        oma_og_slim_out_path,
        oma_og_func_out_path,
        go_levels=[0, 1, 2, 3, 4, 5],
    )

    (
        manual_go_counts,
        go_slim_to_oma_group,
        og_by_go,
    ) = filter_slims_by_count_cooccurence(
        go_parent_count_at_level,
        og_slim_df,
        min_count=MIN_COUNT,
        max_count=MAX_COUNT,
        corr_threshold=CORR_THRESHOLD,
    )
    n_total_oma_groups = og_slim_df["og"].nunique()
    manual_go_counts_df = {
        go: [
            godag[go].name,
            manual_go_counts[go],
            manual_go_counts[go] / n_total_oma_groups,
            (
                set()
                .union(
                    *[
                        go_slim_to_oma_group.get(child, set())
                        for child in godag[go].get_all_children()
                    ]
                )
                .union(go_slim_to_oma_group.get(go, set()))
                if go in godag
                else set()
            ),
        ]
        for go in manual_go_counts.keys()
    }
    # manual_go_counts turn to df
    manual_go_counts_df = pd.DataFrame.from_dict(
        manual_go_counts_df,
        orient="index",
        columns=["go_name", "count", "proportion", "og"],
    )
    manual_go_counts_df.sort_values("count", ascending=False, inplace=True)

    # save manual_go_counts_df to parquet
    manual_go_counts_df.to_parquet(
        "/scratch/suyuelyu/deimm/pp_dayhoff/og_feature/resplit_0331/manual_go_counts_df_w_og.parquet"
    )
    manual_go_counts_df[["go_name", "count", "proportion"]].to_csv(
        "/scratch/suyuelyu/deimm/pp_dayhoff/og_feature/resplit_0331/manual_go_counts_df.csv",
        index=True,
    )

    # add columns in og_slim_df_manual, each colum is a manual go, value is 1 if the og is annotated with this manual go or its children, otherwise 0
    # merge og_by_go with oma_og_features_df to get a final dataframe with og features and manual go annotations, save it to parquet
    # og_by_go turn index to og column
    og_by_go["og"] = og_by_go.index.astype(int)
    og_by_go.reset_index(drop=True, inplace=True)
    final_df = pd.merge(
        oma_og_features_df, og_by_go, on="og", how="left"
    )  # keep all ogs in oma_og_features_df, if an og is not annotated with any manual go, fillna with 0
    final_df.fillna(0, inplace=True)
    # check if any columns contain nan
    nan_cols = final_df.columns[final_df.isna().any()].tolist()
    print(f"Final dataframe shape: {final_df.shape}")
    print(f"Final dataframe columns: {final_df.columns.tolist()}")
    print(f"Columns with NaN values: {nan_cols}")
    save_output_df(
        final_df,
        "/scratch/suyuelyu/deimm/pp_dayhoff/og_feature/resplit_0331/final_og_features_with_manual_go.parquet",
    )


if __name__ == "__main__":
    main()

"""
Run dayhoff and save last protein hidden from decoder layers.
Prepare representation data for probing species info
"""

import pickle
import torch
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
import argparse

seed_everything(3525)
rand_generator = torch.Generator().manual_seed(3525)

# meta_train_no_cluster_leak = pd.read_parquet(
#     "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_train_l1024_all.parquet"
# )
# meta_grouped_train = meta_train_no_cluster_leak.groupby("og").agg(list).reset_index()
# meta_grouped_train = meta_grouped_train.sample(frac=1, random_state=3525).reset_index(
#     drop=True
# )
# # split into 4 chunks and save as parquet
# chunk_size = len(meta_grouped_train) // 5
# for i in range(5):
#     chunk = meta_grouped_train.iloc[i * chunk_size : (i + 1) * chunk_size]
#     chunk.to_parquet(
#         f"/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_train_chunk_{i}.parquet"
#     )

# meta_val = pd.read_parquet(
#     "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_val_l1024_all.parquet"
# )
# meta_grouped_val = meta_val.groupby("og").agg(list).reset_index()

# meta_test = pd.read_parquet(
#     "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_test_l1024_all.parquet"
# )
# meta_grouped_test = meta_test.groupby("og").agg(list).reset_index()
# print(meta_grouped_train.head())
# print(meta_grouped_val.head())
# print(meta_grouped_test.head())

# save to parquet
# meta_grouped_train.to_parquet(
#     "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_train.parquet"
# )
# meta_grouped_val.to_parquet(
#     "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_val.parquet"
# )
# meta_grouped_test.to_parquet(
#     "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet"
# )
# argparse to specify which split to process
parser = argparse.ArgumentParser(description="Save last protein hidden for probing")
parser.add_argument(
    "--train_chunk_idx",
    type=int,
    default=0,
    help="Index of the train chunk to process (0-4)",
)
args = parser.parse_args()


tokenizer_config = {
    "vocab_path": "vocab_UL_ALPHABET_PLUS.txt",
    "allow_unk": False,
}

tokenizer = PureARTokenizer(**tokenizer_config)

# read meta parquet
# meta_grouped_train = pd.read_parquet(
#     f"/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_train_chunk_{args.train_chunk_idx}.parquet"
# )
meta_grouped_val = pd.read_parquet(
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_val.parquet"
)
meta_grouped_test = pd.read_parquet(
    "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet"
)

# load model
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


def get_last_protein_hidden(
    model: torch.nn.Module,
    protein_names: list,
    lins: list,
    seqs: list,
    tokenizer: PureARTokenizer,
    n_max_protein: int = 64,
    layers: list | None = None,
) -> tuple:
    # if above 64, downsample to 64
    n_max_protein = min(n_max_protein, len(seqs))
    idxs = torch.randperm(len(seqs), generator=rand_generator)[:n_max_protein]
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

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states

    # get last protein hidden from all layers
    if layers is None:
        layers = list(range(len(output)))
    last_protein_hiddens = []
    for layer in layers:
        last_protein_hidden = output[layer][0, -last_protein_len:, :]
        last_protein_hiddens.append(last_protein_hidden.cpu())

    del output
    torch.cuda.empty_cache()

    return protein_names, last_protein_hiddens, lins[-1]


# loop through meta_grouped_train, meta_grouped_val, meta_grouped_test
# and save last protein hidden as pickle files
for split, meta_grouped in zip(
    ["val", "test"],
    [meta_grouped_val, meta_grouped_test],
    strict=True,
):
    # for split, meta_grouped in zip(
    #     [f"train_chunk_{args.train_chunk_idx}"],
    #     [meta_grouped_train],
    #     strict=True,
    # ):
    # shuffle the order of meta_grouped
    # meta_grouped = meta_grouped.sample(frac=1, random_state=3525).reset_index(drop=True)
    all_last_protein_hiddens = []
    hidden_chunk_size = 1000
    hidden_file_idx = 0
    for i, row in tqdm(
        meta_grouped.iterrows(), total=len(meta_grouped), desc=f"Processing {split}"
    ):
        try:
            protein_names, last_protein_hiddens, lin = get_last_protein_hidden(
                pretrained_jamba_model,
                row["protein"],
                row["taxid"],
                row["seq"],
                tokenizer,
                layers=[16, 24, 28, -1],  # only get last layer hidden
            )

            all_last_protein_hiddens.append(
                {
                    "og": row["og"],
                    "protein_names": protein_names,
                    "last_protein_hiddens": last_protein_hiddens,
                    "lin": lin,
                }
            )
            if (i + 1) % hidden_chunk_size == 0:
                print(f"Processed {i + 1}/{len(meta_grouped)} {split} samples")
                with open(
                    f"/scratch/suyuelyu/deimm/data/oma/oma_probe_last_protein_hidden_lyr16_24_28_32_{split}_{hidden_file_idx}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(all_last_protein_hiddens, f)
                all_last_protein_hiddens = []
                hidden_file_idx += 1
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"Error processing {row['og']}: {e}")
            continue

    # save remaining samples
    if len(all_last_protein_hiddens) > 0:
        with open(
            f"/scratch/suyuelyu/deimm/data/oma/oma_probe_last_protein_hidden_lyr16_24_28_32_{split}_{hidden_file_idx}.pkl",
            "wb",
        ) as f:
            pickle.dump(all_last_protein_hiddens, f)

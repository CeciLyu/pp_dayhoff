"""
Run dayhoff and add steering vector from saved probe to last hidden layer.
Check if PPL decrease when steer with right vs wrong taxon label.
"""

import pickle
import torch
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

# RANKS_TO_PROBE = ["domain", "phylum", "class", "order", "family", "genus", "species"]
RANKS_TO_PROBE = ["class"]
PROBE_PATH = Path("/scratch/suyuelyu/deimm/results/probe_taxon/")
SEED = 3525
STEER_GRAD_ALPHA = [0.1, 0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
seed_everything(SEED)
rand_generator = torch.Generator().manual_seed(SEED)


def load_dayhoff_model_tokenizer() -> tuple[torch.nn.Module, PureARTokenizer]:
    """Load the pretrained Jamba model and tokenizer for Dayhoff generation."""
    tokenizer_config = {
        "vocab_path": "vocab_UL_ALPHABET_PLUS.txt",
        "allow_unk": False,
    }

    tokenizer = PureARTokenizer(**tokenizer_config)
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
    return pretrained_jamba_model, tokenizer


def load_linear_probs(probe_path: str | Path) -> dict[str, torch.Tensor | dict]:
    """Load the linear probe weights for each taxonomic rank."""
    print(f"Loading probe from {probe_path}...")
    with open(probe_path, "rb") as f:
        probe_data = pickle.load(f)
    # probe_data is a dict, W: is the linear layer weights: (hidden_dim, n_classes)
    linear_weights = torch.from_numpy(probe_data["W"]).float().to("cuda")
    # dict of dicts: {rank: {species_id: rank_label}}
    species_to_rank = probe_data["rank_mapping"]
    # dict: {rank_label: class_idx}
    rank_to_classidx = {lab: idx for idx, lab in enumerate(probe_data["classes"])}
    intercept = torch.from_numpy(probe_data["intercept"]).float().to("cuda")
    return {
        "linear_weights": linear_weights,
        "species_to_rank": species_to_rank,
        "rank_to_classidx": rank_to_classidx,
        "intercept": intercept,
    }


@torch.inference_mode()
def get_last_protein_hidden_w_logits(
    model: torch.nn.Module,
    protein_names: list,
    lins: list,
    seqs: list,
    tokenizer: PureARTokenizer,
    n_max_protein: int = 64,
    layers: list | None = None,
) -> tuple:
    """Get the hidden states of the last protein in the input sequence and logits."""
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

    output = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = output.hidden_states  # tuple of (batch_size, seq_len, hidden_dim)
    logits = output.logits[
        0, -(last_protein_len + 1) : -1, :
    ]  # (batch_size, seq_len, vocab_size)

    # get last protein hidden from all layers
    if layers is None:
        layers = list(range(len(hidden_states)))
    last_protein_hiddens = []
    for layer in layers:
        last_protein_hidden = hidden_states[layer][0, -(last_protein_len + 1) : -1 :, :]
        last_protein_hiddens.append(last_protein_hidden.cpu())

    del output, hidden_states, input_ids
    torch.cuda.empty_cache()

    return protein_names, last_protein_hiddens, logits.cpu(), lins[-1], seqs[-1]


@torch.inference_mode()
def get_logits_with_steer(
    model: torch.nn.Module,
    last_protein_hidden: torch.Tensor,
    linear_weights: torch.Tensor,
    intercept: torch.Tensor | None,
    right_species_id: int,
    species_to_rank: dict,
    rank_to_classidx: dict,
    alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int] | tuple[None, None, None, None]:
    """Steer with right taxon and a random wrong taxon, and get the new logits."""
    # get rank label for the right species id
    right_logits = None
    wrong_logits = None
    right_rank = species_to_rank.get(right_species_id, -1)
    right_rank_idx = rank_to_classidx.get(right_rank, -1)
    if right_rank_idx == -1:
        # print a warning and return None
        print(f"  WARNING: species_id {right_species_id} not found in probe mapping")
        return None, None, None, None

    # randomly select an item from rank_to_classidx that is not the right label
    wrong_rank = right_rank
    while wrong_rank == right_rank:
        wrong_rank = random.choice(list(rank_to_classidx.keys()))
    wrong_rank_idx = rank_to_classidx[wrong_rank]

    right_steer = linear_weights[:, right_rank_idx]  # (hidden_dim,)
    wrong_steer = linear_weights[:, wrong_rank_idx]  # (hidden_dim,)
    last_protein_hidden = last_protein_hidden.to("cuda").float()

    if alpha is not None:
        # predict class prob
        with torch.no_grad():
            prob = torch.softmax(
                torch.matmul(last_protein_hidden, linear_weights) + intercept,
                dim=-1,
            )
            # gradient
            e_target = torch.zeros_like(prob)
            e_target[:, right_rank_idx] = 1.0
            grad = (prob - e_target) @ linear_weights.t()  # (seq_len, hidden_dim)
            # adaptive alpha
            right_steer = -alpha * grad
            wrong_steer = alpha * grad

    # steer the hidden state
    right_steered_hidden = last_protein_hidden + right_steer * 2
    wrong_steered_hidden = last_protein_hidden + wrong_steer * 2

    # layernorm the steered hidden states
    right_steered_hidden = model.model.final_layernorm(right_steered_hidden)
    wrong_steered_hidden = model.model.final_layernorm(wrong_steered_hidden)
    # get new logits
    right_logits = model.lm_head(
        right_steered_hidden.unsqueeze(0).half()
    ).cpu()  # (1, seq_len, vocab_size)
    wrong_logits = model.lm_head(wrong_steered_hidden.unsqueeze(0).half()).cpu()

    return right_logits, wrong_logits, wrong_rank, right_rank


@torch.inference_mode()
def calc_ppl(
    logits: torch.Tensor, target_seq: str, tokenizer: PureARTokenizer
) -> tuple[torch.Tensor, float]:
    """Calculate the perplexity of the target sequence given the logits."""
    target_ids = tokenizer.tokenize_protein(target_seq).to("cpu")  # (seq_len,)
    logits = logits.to("cpu")  # (seq_len, vocab_size)
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        reduction="none",
    )  # (seq_len,)
    ppl = torch.exp(ce)  # (seq_len,)
    mean_ppl = torch.exp(ce.mean()).item()
    return ppl, mean_ppl


def check_probe_vs_lmhead(
    model,
    linear_weights,
):
    W_lm = model.lm_head.weight.data.float()  # [40, 1280]

    # SVD of LM head to get its active subspace
    U, S, Vt = torch.linalg.svd(W_lm, full_matrices=False)  # Vt: [40, 1280]
    lm_subspace = Vt  # 40-dimensional subspace the LM head uses

    # Project each probe class direction onto the LM head subspace
    for c in range(linear_weights.size(1)):
        v = torch.tensor(linear_weights[:, c]).float()
        v_norm = v / v.norm()

        # Projection onto LM head subspace
        proj = lm_subspace @ v_norm  # [40]
        frac_in_subspace = proj.norm() ** 2  # fraction of variance in LM subspace
        frac_orthogonal = 1.0 - frac_in_subspace

        print(
            f"Class {c}: {frac_in_subspace:.4f} in LM subspace, "
            f"{frac_orthogonal:.4f} orthogonal"
        )


def main() -> None:
    meta_grouped_test = pd.read_parquet(
        "/scratch/suyuelyu/deimm/data/oma/oma_probe_meta_grouped_test.parquet"
    )
    model, tokenizer = load_dayhoff_model_tokenizer()
    all_rank_probes = {}
    for rank in RANKS_TO_PROBE:
        print(f"Loading probe for {rank}...")
        # probe_path = probe_{rank}_data.pkl
        # probe_path = PROBE_PATH.joinpath(f"probe_{rank}_data.pkl")
        # /scratch/suyuelyu/deimm/results/probe_taxon/class_ce_mmap/probe_class_data.pkl
        probe_path = PROBE_PATH.joinpath(f"{rank}_ce_mmap", f"probe_{rank}_data.pkl")
        all_rank_probes[rank] = load_linear_probs(probe_path)

        check_probe_vs_lmhead(model, all_rank_probes[rank]["linear_weights"])
        breakpoint()
    # get last hidden and logits
    # save initial conditions before steering
    condition_output_dir = PROBE_PATH.joinpath(
        "steer_ppl", f"initial_conditions_seed{SEED}.pkl"
    )
    condition_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not condition_output_dir.exists():
        conditions = {}
        for _, row in tqdm(
            meta_grouped_test.iterrows(),
            total=len(meta_grouped_test),
            desc="Processing unsteered",
        ):
            protein_names, last_protein_hiddens, logits, species_id, seq = (
                get_last_protein_hidden_w_logits(
                    model=model,
                    protein_names=row["protein"],
                    lins=row["taxid"],
                    seqs=row["seq"],
                    tokenizer=tokenizer,
                    layers=[-1],  # only get last layer hidden
                )
            )
            unsteered_ppls, mean_unsteered_ppl = calc_ppl(logits, seq, tokenizer)
            conditions[row["og"]] = {
                "species_id": species_id,
                "seq": seq,
                "protein_names": protein_names,
                "last_protein_hidden": last_protein_hiddens[0],
                "unsteered_logits": logits,
                "unsteered_ppls": unsteered_ppls,
                "mean_unsteered_ppl": mean_unsteered_ppl,
            }
        with open(condition_output_dir, "wb") as f:
            pickle.dump(conditions, f)
    else:
        with open(condition_output_dir, "rb") as f:
            conditions = pickle.load(f)

    for rank in RANKS_TO_PROBE:
        # steer and get new logits
        print(f"Steering with {rank} probe...")
        rank_probe = all_rank_probes[rank]
        linear_weights = rank_probe["linear_weights"]
        intercept = rank_probe.get("intercept", None)
        species_to_rank = rank_probe["species_to_rank"]
        rank_to_classidx = rank_probe["rank_to_classidx"]

        output_dir = PROBE_PATH.joinpath(
            "steer_ppl", f"{rank}_steered_conditions_seed{SEED}.pkl"
        )
        for i, row in tqdm(
            meta_grouped_test.iterrows(),
            total=len(meta_grouped_test),
            desc=f"Steering with {rank} probe",
        ):
            og = row["og"]
            condition = conditions[og]
            species_id = condition["species_id"]
            last_protein_hidden = condition["last_protein_hidden"]
            right_logits, wrong_logits, wrong_rank, right_rank = get_logits_with_steer(
                model=model,
                last_protein_hidden=last_protein_hidden,
                linear_weights=linear_weights,
                intercept=intercept,
                right_species_id=species_id,
                species_to_rank=species_to_rank,
                rank_to_classidx=rank_to_classidx,
                alpha=None,
            )
            if right_logits is None or wrong_logits is None:
                continue

            conditions[og][f"{rank}_uniform_steer"] = {}
            conditions[og][f"{rank}_uniform_steer"]["right_logits"] = right_logits
            conditions[og][f"{rank}_uniform_steer"]["wrong_logits"] = wrong_logits
            conditions[og][f"{rank}_uniform_steer"]["wrong"] = wrong_rank
            conditions[og][f"{rank}_uniform_steer"]["right"] = right_rank

            right_ppls, mean_right_ppl = calc_ppl(
                right_logits, condition["seq"], tokenizer
            )
            wrong_ppls, mean_wrong_ppl = calc_ppl(
                wrong_logits, condition["seq"], tokenizer
            )
            conditions[og][f"{rank}_uniform_steer"]["right_ppls"] = right_ppls
            conditions[og][f"{rank}_uniform_steer"]["wrong_ppls"] = wrong_ppls
            conditions[og][f"{rank}_uniform_steer"]["mean_right_ppl"] = mean_right_ppl
            conditions[og][f"{rank}_uniform_steer"]["mean_wrong_ppl"] = mean_wrong_ppl
            print(
                f"  {og}: right_rank={right_rank}, wrong_rank={wrong_rank}, uniform_steer, "
                f"unsteered_ppl={condition['mean_unsteered_ppl']:.2f}, "
                f"right_steered_ppl={mean_right_ppl:.2f}, "
                f"wrong_steered_ppl={mean_wrong_ppl:.2f}",
            )
            for alpha in STEER_GRAD_ALPHA:
                right_logits, wrong_logits, wrong_rank, right_rank = (
                    get_logits_with_steer(
                        model=model,
                        last_protein_hidden=last_protein_hidden,
                        linear_weights=linear_weights,
                        intercept=intercept,
                        right_species_id=species_id,
                        species_to_rank=species_to_rank,
                        rank_to_classidx=rank_to_classidx,
                        alpha=alpha,
                    )
                )
                if right_logits is None or wrong_logits is None:
                    continue

                conditions[og][f"{rank}_{alpha}"] = {}
                conditions[og][f"{rank}_{alpha}"]["right_logits"] = right_logits
                conditions[og][f"{rank}_{alpha}"]["wrong_logits"] = wrong_logits
                conditions[og][f"{rank}_{alpha}"]["wrong"] = wrong_rank
                conditions[og][f"{rank}_{alpha}"]["right"] = right_rank

                # calculate PPL for right steer, and wrong steer
                right_ppls, mean_right_ppl = calc_ppl(
                    right_logits, condition["seq"], tokenizer
                )
                wrong_ppls, mean_wrong_ppl = calc_ppl(
                    wrong_logits, condition["seq"], tokenizer
                )
                conditions[og][f"{rank}_{alpha}"]["right_ppls"] = right_ppls
                conditions[og][f"{rank}_{alpha}"]["wrong_ppls"] = wrong_ppls
                conditions[og][f"{rank}_{alpha}"]["mean_right_ppl"] = mean_right_ppl
                conditions[og][f"{rank}_{alpha}"]["mean_wrong_ppl"] = mean_wrong_ppl

                print(
                    f"  {og}: right_rank={right_rank}, wrong_rank={wrong_rank}, alpha={alpha}, "
                    f"unsteered_ppl={condition['mean_unsteered_ppl']:.2f}",
                    f"right_steered_ppl={mean_right_ppl:.2f}",
                    f"wrong_steered_ppl={mean_wrong_ppl:.2f}",
                )

            # save every 100 iteration
            if i % 100 == 0:
                with open(output_dir, "wb") as f:
                    pickle.dump(conditions, f)

        with open(output_dir, "wb") as f:
            pickle.dump(conditions, f)

    return None


if __name__ == "__main__":
    main()

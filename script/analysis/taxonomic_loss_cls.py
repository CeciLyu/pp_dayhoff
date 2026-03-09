# === Setup ===

# taxonomy_tree is a dict: can be found in DEFAULT_TAXONOMY_MAPPING
# e.g. taxonomy_tree[species_id] = {rank1: rank1_id, rank2: rank2_id, ...}

# For each taxonomic level, build a binary aggregation matrix
# that sums species probs into that level's probs
# shape: (num_classes_at_level, num_species)

agg_matrices = {}
for level in ["genus", "family", "order", "class", "phylum", "domain"]:
    n_groups = num_classes_at[level]
    M = zeros(n_groups, num_species)
    for species_idx in range(num_species):
        group_idx = taxonomy_tree[species_idx][level]
        M[group_idx, species_idx] = 1.0
    agg_matrices[level] = M  # fixed, not learnable


# === Model ===

# Single linear probe: (embedding_dim, num_species)
W = Linear(1280, num_species)


# === Forward + Loss ===

def taxonomic_loss(embeddings, species_labels):
    # embeddings: (batch, seq_len, 1280) — per-residue
    # species_labels: (batch,) — same label for every residue in a protein

    # flatten residues: (batch * seq_len, 1280)
    x = embeddings.reshape(-1, 1280)

    # repeat labels per residue: (batch * seq_len,)
    labels = species_labels.repeat_interleave(seq_len)

    # species logits and probs
    logits = W(x)                          # (N, num_species)
    species_probs = softmax(logits, dim=-1)

    # species-level CE
    loss = cross_entropy(logits, labels)

    # aggregate up the tree and add CE at each level
    for level in ["genus", "family", "order", "class", "phylum", "domain"]:
        M = agg_matrices[level]                          # (num_groups, num_species)
        level_probs = species_probs @ M.T                # (N, num_groups)
        level_labels = taxonomy_tree.map(labels, level)  # map species labels to this level
        level_log_probs = log(level_probs + eps)
        loss += weight[level] * nll_loss(level_log_probs, level_labels)

    return loss


# === Evaluation ===

# species accuracy: argmax of logits
# genus accuracy:   argmax of (species_probs @ M_genus.T)
# family accuracy:  argmax of (species_probs @ M_family.T)
# ... etc
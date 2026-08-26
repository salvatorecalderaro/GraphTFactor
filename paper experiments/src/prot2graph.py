import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from numba import njit

seed = 2025
aminoacidi = 'ACDEFGHIKLMNPQRSTVWYX'

AA_TO_INT = {aa: i for i, aa in enumerate(aminoacidi)}

def encode_sequence(sequence):
    sequence = check_input(sequence)
    return np.array([AA_TO_INT[c] for c in sequence], dtype=np.int32)





def check_input(sequence):
    """
    Check that a given sequence contains only valid amino acids.

    Args:
        sequence (str): The sequence to be checked.

    Returns:
        str: The sequence with invalid amino acids replaced with 'X'.
    """
    sequence = sequence.upper()
    return ''.join([aa if aa in aminoacidi else 'X' for aa in sequence])


@njit
def segment_cost(seq, start, end):

    counts = np.zeros(21, dtype=np.int32)

    for i in range(start, end):
        counts[seq[i]] += 1

    n = end - start
    cost = 0.0

    for c in counts:
        if c > 0:
            p = c / n
            cost -= c * np.log(p)

    return cost



@njit
def mdl_segmentation_numba(seq):

    N = len(seq)

    lam = np.log(N)

    DP = np.full(N + 1, np.inf)
    prev = np.full(N + 1, -1)

    DP[0] = 0.0

    for i in range(1, N + 1):

        best_cost = DP[i - 1] + segment_cost(seq, i - 1, i) + lam
        best_prev = i - 1

        for j in range(i - 1):

            c = DP[j] + segment_cost(seq, j, i) + lam

            if c < best_cost:
                best_cost = c
                best_prev = j

        DP[i] = best_cost
        prev[i] = best_prev

    return prev




def get_token_embeddings(model, alphabet, n_layer, device, sequence):
    """
    Get the token embeddings for a given sequence.

    Args:
        model (torch.nn.Module): A pre-trained ESM model.
        alphabet (esm.Alphabet): The alphabet corresponding to the ESM model.
        n_layer (int): The number of layer to obtain the token embeddings from.
        device (torch.device): The device to be used for PyTorch computations.
        sequence (str): The sequence to obtain the token embeddings for.

    Returns:
        torch.Tensor: The token embeddings for the given sequence (L, d).

    """
    batch_converter = alphabet.get_batch_converter()
    with torch.no_grad():
        _, _, tokens = batch_converter([("", sequence)])
        tokens = tokens.to(device)
        out = model(tokens, repr_layers=[n_layer])
        reps = out["representations"][n_layer]
        reps = reps[0, 1:-1]  # remove CLS/EOS
    return reps.cpu()


def region_embedding(tokens, start, end):
    w = torch.linspace(0.5, 1.0, end - start)
    w = w / w.sum()
    return (tokens[start:end] * w[:, None]).sum(dim=0)


def get_region_embeddings(model, alphabet, n_layer, device, sequence, segments):
    """
    Get the region embeddings for a given sequence and segments.

    Args:
        model (torch.nn.Module): A pre-trained ESM model.
        alphabet (esm.Alphabet): The alphabet corresponding to the ESM model.
        n_layer (int): The number of layer to obtain the token embeddings from.
        device (torch.device): The device to be used for PyTorch computations.
        sequence (str): The sequence to obtain the region embeddings for.
        segments (list[tuple[int, int]]): A list of tuples containing the start and end indices of the segments.

    Returns:
        torch.Tensor: The region embeddings for the given sequence and segments (num_segments, d).
    """
    tokens = get_token_embeddings(model, alphabet, n_layer, device, sequence)
    E = []
    for start, end in segments:
        E.append(region_embedding(tokens, start, end))
    return torch.stack(E)  # (num_segments, d)


def local_differences(E):
    """
    Compute the local differences of the region embeddings.

    If the number of region embeddings is greater than 1, compute the
    local differences by subtracting the previous region embedding from the
    current one. If the number of region embeddings is 1, return the
    single region embedding as is.

    Args:
        E (torch.Tensor): The region embeddings (num_segments, d).

    Returns:
        torch.Tensor: The local differences of the region embeddings (num_segments - 1, d).
    """
    if E.size(0) > 1:
        return E[1:] - E[:-1]
    return E 

def mdl_segmentation(sequence):

    seq = encode_sequence(sequence)

    prev = mdl_segmentation_numba(seq)

    segments = []

    i = len(seq)

    while i > 0:

        j = int(prev[i])

        segments.append((j, i))

        i = j

    return segments[::-1]



def create_graph(sequence, esm_model, alphabet, n_layer, device, y, percentile=0.7,return_segments=False):
    """
    Create a graph from a protein sequence using ESM embeddings.

    Nodes represent segmented regions of the sequence (via MDL),
    edges connect regions with strong similarity (adaptive percentile threshold).

    Args:
        sequence (str): Protein sequence.
        esm_model (torch.nn.Module): Pre-trained ESM model.
        alphabet (esm.Alphabet): Corresponding alphabet.
        n_layer (int): Layer from which to extract token embeddings.
        device (torch.device): Device for PyTorch computations.
        y (int): Graph label.
        percentile (float, optional): Percentile for adaptive similarity threshold (default 0.8).

    Returns:
        torch_geometric.data.Data: Graph object with node features, edges, weights, and label.
        list[tuple[int, int]] (optional): List of segments if return_segments is True.
    """

    # --- Step 1: sanitize sequence
    sequence = check_input(sequence)

    # --- Step 2: segment sequence (MDL)
    segments = mdl_segmentation(sequence)
    
    

    # --- Step 3: compute region embeddings
    E = get_region_embeddings(esm_model, alphabet, n_layer, device, sequence, segments)

    # --- Step 4: compute local differences (optional)
    E = local_differences(E)

    # --- Raise error if no nodes
    if E is None or E.size(0) == 0:
        raise ValueError("Void graph detected: zero nodes after local differences.")

    # --- Step 5: normalize embeddings
    E = F.normalize(E, dim=1)

    # --- Step 6: similarity matrix
    sim = torch.matmul(E, E.T)

    # --- Step 7: handle special cases
    if E.size(0) == 1:
        # single node fallback: self-loop
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0])
    else:
        # remove self-loops for threshold calculation
        sim.fill_diagonal_(0)

        # fallback if no positive similarity exists
        if (sim > 0).sum() == 0:
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            edge_weight = torch.tensor([1.0])
        else:
            # --- Step 8: adaptive threshold based on percentile
            threshold = torch.quantile(sim[sim > 0], percentile)
            A = (sim >= threshold).float()

            # remove self-loops
            A.fill_diagonal_(0)

            # fallback if no edges survive threshold
            if A.sum() == 0:
                edge_index = torch.tensor([[0],[0]], dtype=torch.long)
                edge_weight = torch.tensor([1.0])
            else:
                # --- Step 9: create edge_index and edge_weight
                edge_index = A.nonzero(as_tuple=False).T
                edge_weight = sim[A == 1]

    # --- Step 10: construct PyG graph
    y  = np.array([y], dtype=np.int64)
    graph = Data(
        x=E,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor(y, dtype=torch.long)
    )

    if return_segments:
        return graph, segments
    return graph
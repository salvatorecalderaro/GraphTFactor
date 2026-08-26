import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from numba import njit

aminoacidi = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_INT = {aa: i for i, aa in enumerate(aminoacidi)}

seed = 2026

def check_input(sequence):
    """
    Check and preprocess the input protein sequence.
    Args:
        sequence (str): The input protein sequence.
    Returns:
        str: The preprocessed protein sequence with invalid characters replaced by 'X'.
    """
    sequence = sequence.upper()
    return ''.join([aa if aa in aminoacidi else 'X' for aa in sequence])

def encode_sequence(sequence):
    sequence = check_input(sequence)
    return np.array([AA_TO_INT[c] for c in sequence], dtype=np.int32)

@njit
def segment_cost(seq, start, end):
    """
    Calculate the cost of a segment of the sequence using the Minimum Description Length (MDL) principle.
    Args:
        seq (np.ndarray): The encoded protein sequence.
        start (int): The starting index of the segment.
        end (int): The ending index of the segment.
    Returns:
        float: The cost of the segment.
    """
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
    """
    Perform Minimum Description Length (MDL) segmentation on the encoded protein sequence.
    Args:
        seq (np.ndarray): The encoded protein sequence.
    Returns:
        np.ndarray: An array containing the previous segment indices for each position in the sequence.
    """

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


def get_random_token_embeddings(sequence, aa_embeddings):
    """
    Get random token embeddings for the input protein sequence using the provided amino acid embeddings.
    Args:
        sequence (str): The input protein sequence.
        aa_embeddings (dict): A dictionary mapping amino acids to their corresponding embeddings.
    Returns:
        torch.Tensor: A tensor containing the embeddings for each amino acid in the sequence.
    """
    sequence = check_input(sequence)
    return torch.stack([aa_embeddings[aa]for aa in sequence])
    
def region_embedding(tokens, start, end):
    """
    Calculate the embedding for a specific region of the token embeddings using a weighted average.
    
    Args:
        tokens (torch.Tensor): The token embeddings for the entire sequence.
        start (int): The starting index of the region.
        end (int): The ending index of the region.
    Returns:
        torch.Tensor: The embedding for the specified region.
    """
    w = torch.linspace(0.5, 1.0, end - start)
    w = w / w.sum()
    return (tokens[start:end] * w[:, None]).sum(dim=0)

def get_random_region_embeddings(sequence, segments, aa_embeddings):
    """
    Get random region embeddings for the specified segments of the input protein sequence using the provided amino acid embeddings.
    Args:
        sequence (str): The input protein sequence.
        segments (list): A list of tuples representing the start and end indices of the segments.
        aa_embeddings (dict): A dictionary mapping amino acids to their corresponding embeddings.
    Returns:
        torch.Tensor: A tensor containing the embeddings for each specified region.
    """
    tokens = get_random_token_embeddings(sequence, aa_embeddings)
    E = []
    for start, end in segments:
        E.append(region_embedding(tokens, start, end))
    return torch.stack(E)

def local_differences(E):
    """
    Calculate the local differences of the embeddings to capture local variations.
    Args:
        E (torch.Tensor): A tensor containing the embeddings for each region.
    Returns:
        torch.Tensor: A tensor containing the local differences of the embeddings.
    """
    if E.size(0) > 1:
        return E[1:] - E[:-1]
    return E

def mdl_segmentation(sequence):
    """
    Perform Minimum Description Length (MDL) segmentation on the input protein sequence.
    Args:
        sequence (str): The input protein sequence.
    Returns:
        list: A list of tuples representing the start and end indices of the segments.
    """
    seq = encode_sequence(sequence)
    prev = mdl_segmentation_numba(seq)
    segments = []
    i = len(seq)
    while i > 0:
        j = int(prev[i])
        segments.append((j, i))
        i = j
    return segments[::-1]


def create_graph(sequence, y, aa_embeddings, percentile=0.7, return_segments=False):
    """
    Create a graph representation of the input protein sequence based on its embeddings and similarity.
    Args:
        sequence (str): The input protein sequence.
        y (int): The label for the graph (e.g., 0 or 1).
        aa_embeddings (dict): A dictionary mapping amino acids to their corresponding embeddings.
        percentile (float): The percentile threshold for determining edges based on similarity.
        return_segments (bool): Whether to return the segments along with the graph.
    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    sequence = check_input(sequence)
    segments = mdl_segmentation(sequence)
    E = get_random_region_embeddings(sequence,segments,aa_embeddings)
    E = local_differences(E)
    E = F.normalize(E, p=2, dim=1)
    sim = torch.matmul(E, E.T)
    
    if E.size(0) == 1:
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0])
    else:
        sim.fill_diagonal_(0)
        if (sim > 0).sum() == 0:
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
            edge_weight = torch.tensor([1.0])
        else:
            threshold = torch.quantile(sim[sim > 0], percentile)
            A = (sim >= threshold).float()
            A.fill_diagonal_(0)
            if A.sum() == 0:
                edge_index = torch.tensor([[0],[0]], dtype=torch.long)
                edge_weight = torch.tensor([1.0])
            else:
                edge_index = A.nonzero(as_tuple=False).T
                edge_weight = sim[A == 1]

    graph = Data(
        x=E,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([y], dtype=torch.long)
    )

    if return_segments:
        subseqs = [sequence[start:end] for start, end in segments]
        return graph, subseqs
    return graph
    
    
    
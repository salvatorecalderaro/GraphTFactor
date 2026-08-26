import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

SEED = 2026

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}



def check_gpu_memory(device,min_free_gb=2.0):
    """
    Check if the GPU has enough free memory. If not, clear the cache and raise an error.
    Args:
        device (torch.device): The device to check for available memory.
        min_free_gb (float): Minimum free memory in GB required to proceed.
    Raises:
        RuntimeError: If the available GPU memory is less than the specified minimum.
    """
    if not torch.cuda.is_available():
        return
    if device.type != "cuda":
        return


    free, total = torch.cuda.mem_get_info(device)

    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        torch.cuda.empty_cache()

        raise RuntimeError( f"Not enough GPU memory " f"({free_gb:.2f} GB free)")



def check_input(sequence):
    """
    Check and preprocess the input protein sequence.
    Args:
        sequence (str): The input protein sequence.
    Returns:
        str: The preprocessed protein sequence with invalid characters replaced by 'X'.
    """

    if sequence is None:
        raise ValueError("Sequence is None")


    sequence = sequence.upper().strip()


    if len(sequence) == 0:
        raise ValueError("Empty sequence")


    return "".join(
        aa if aa in AMINO_ACIDS else "X"
        for aa in sequence
    )



def encode_sequence(sequence):
    """
    Encode the protein sequence into a numerical representation.
    Args:
        sequence (str): The input protein sequence.
    Returns:
        np.ndarray: An array of integers representing the encoded sequence."""
    sequence = check_input(sequence)
    return np.asarray(
        [
            AA_TO_INT[a]
            for a in sequence
        ],
        dtype=np.int32
    )



def get_token_embeddings(model,alphabet,n_layer,device,sequence):
    """
    Get the token embeddings for a given protein sequence using the ESM model.
    Args:
        model: The ESM model.
        alphabet: The ESM model's alphabet.
        n_layer (int): The layer number from which to extract embeddings.
        device (torch.device): The device to run the model on.
        sequence (str): The input protein sequence.
    Returns:
        torch.Tensor: The token embeddings for the input sequence.
    """

    batch_converter = alphabet.get_batch_converter()


    with torch.inference_mode():
        _, _, tokens = batch_converter([("", sequence)])
        tokens = tokens.to(device)



        try:
            check_gpu_memory(device)
            output = model(tokens,repr_layers=[n_layer])

        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM during ESM embedding")
            torch.cuda.empty_cache()
            raise RuntimeError( "ESM failed due to GPU memory")



        embeddings = output["representations"][n_layer]
        embeddings = embeddings[0,1:-1]
    return embeddings.cpu()



def region_embedding(tokens,start,end):
    """
    Compute the embedding for a specific region of the token embeddings.
    Args:
        tokens (torch.Tensor): The token embeddings.
        start (int): The starting index of the region.
        end (int): The ending index of the region.
    Returns:
        torch.Tensor: The embedding for the specified region.
    """
    region = tokens[start:end]
    length = region.shape[0]
    weights=torch.linspace(0.5,1.0,length)
    weights /= weights.sum()
    return (region *weights[:,None]).sum(dim=0)



def get_region_embeddings(model,alphabet,n_layer,device,sequence,segments):
    """
    Get the embeddings for specified regions of a protein sequence.
    Args:
        model: The ESM model.
        alphabet: The ESM model's alphabet.
        n_layer (int): The layer number from which to extract embeddings.
        device (torch.device): The device to run the model on.
        sequence (str): The input protein sequence.
        segments (list of tuple): List of (start, end) indices for regions.
    Returns:
        torch.Tensor: The embeddings for the specified regions.
    """
    tokens=get_token_embeddings(model,alphabet,n_layer,device,sequence)
    embeddings=[]


    for start,end in segments:
        reg_emb = region_embedding(tokens,start,end)
        embeddings.append(reg_emb)

    return torch.stack(embeddings)


def local_differences(E):
    """
    Compute the local differences of the embeddings.
    Args:
        E (torch.Tensor): The input embeddings.
    Returns:
        torch.Tensor: The local differences of the embeddings.
    """
    if E.shape[0] > 1:
        return E[1:] - E[:-1]
    return E




def random_log_segmentation(sequence):
    """
    Perform random segmentation of a protein sequence into regions based on a logarithmic scale.
    Args:
        sequence (str): The input protein sequence.
    Returns:
        list of tuple: A list of (start, end) indices representing the segments.
    """

    rng=random.Random(SEED)
    L=len(sequence)
    if L <= 2:
        return [(0,L)]
    
    n_segments = np.ceil(np.sqrt(len(sequence))).astype(int)
    n_segments=min(n_segments,L)
    cuts=sorted(rng.sample(range(1,L),n_segments-1))



    segments=[]
    start=0
    for end in cuts:
        segments.append((start,end))
        start=end



    segments.append((start,L))
    return segments



def build_similarity_graph(E,percentile=0.7):
    """
    Build a similarity graph based on the embeddings and a specified percentile threshold.
    Args:
        E (torch.Tensor): The input embeddings.
        percentile (float): The percentile threshold for determining edges based on similarity.
    Returns:
        tuple: A tuple containing the edge indices and edge weights of the graph.
    """
    if E.shape[0] == 1:

        return (
            torch.tensor(
                [[0],[0]],
                dtype=torch.long
            ),
            torch.tensor(
                [1.0]
            )
        )



    similarity=torch.matmul(E,E.T)
    similarity.fill_diagonal_(0)
    positive=similarity[similarity > 0]

    if positive.numel()==0:

        return (
            torch.tensor(
                [[0],[0]],
                dtype=torch.long
            ),
            torch.tensor(
                [1.0]
            )
        )



    threshold=torch.quantile(positive,percentile)



    adjacency=(similarity >= threshold).float()



    adjacency.fill_diagonal_(0)



    edges=adjacency.nonzero( as_tuple=False)



    if edges.numel()==0:

        return (
            torch.tensor(
                [[0],[0]],
                dtype=torch.long
            ),
            torch.tensor(
                [1.0]
            )
        )



    edge_index=edges.T



    edge_weight=similarity[adjacency == 1]

    return edge_index,edge_weight


def create_graph(sequence,esm_model,alphabet,n_layer,device, y,percentile=0.7,return_segments=False):
    """
    Create a graph representation of the input protein sequence based on its embeddings and similarity.
    Args:
        sequence (str): The input protein sequence.
        esm_model: The ESM model.
        alphabet: The ESM model's alphabet.
        n_layer (int): The layer number from which to extract embeddings.
        device (torch.device): The device to run the model on.
        y (int): The label for the graph (e.g., 0 or 1).
        percentile (float): The percentile threshold for determining edges based on similarity.
        return_segments (bool): Whether to return the segments along with the graph.
    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    segments=random_log_segmentation(sequence)
    E=get_region_embeddings(
        esm_model,
        alphabet,
        n_layer,
        device,
        sequence,
        segments
    )



    E=local_differences(E)
    E=F.normalize(E,dim=1)
    edge_index,edge_weight=build_similarity_graph(E,percentile)



    graph=Data(
        x=E,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor(
            [y],
            dtype=torch.long
        )
    )



    if torch.cuda.is_available():
        torch.cuda.empty_cache()



    if return_segments:
        return graph,segments

    return graph
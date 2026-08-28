import platform
import torch
import streamlit as st
import cpuinfo
import esm 
from model import GraphTFactor
import networkx as nx
from pathlib import Path
dropout = 0.2

def identify_device():
    """
    Identify the available device for PyTorch computations (CPU, CUDA, or MPS).
    Returns:
        tuple: A tuple containing the identified device and its name.
    """
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name

@st.cache_resource
def load_model(n_layers, device):
    """
    Load an ESM model.

    Args:
        n_layers (int): The number of layers in the ESM model.
        device (torch.device): The device to be used for PyTorch computations.

    Returns:
        tuple[torch.nn.Module, esm.Alphabet]: A tuple containing the ESM model and its corresponding alphabet.

    Raises:
        ValueError: If the number of layers is not supported by the ESM model.
    """
    if n_layers == 6:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif n_layers == 12:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif n_layers == 30:
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    else:
        raise ValueError("Unsupported ESM model")

    model = model.to(device)
    model.eval()
    return model, alphabet

def load_model_from_file(org,esm_model,in_channels,device):
    """
    Load a PyTorch model from a file.

    Args:
        model_path (str): The path to the model file.
        device (torch.device): The device to be used for PyTorch computations.
    """
    
    net = GraphTFactor(in_channels=in_channels, dropout=dropout)
    net = net.to(device)
    path = f"checkpoints/{org}/GraphTF_{esm_model}.pth"
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def create_nx_graph(graph):
    G = nx.Graph()
    edges = graph.edge_index.cpu().numpy()
    for i in range(edges.shape[1]):
        G.add_edge(int(edges[0, i]), int(edges[1, i]))
    return G
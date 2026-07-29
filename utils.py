import platform
import torch
import cpuinfo
import esm 
from GraphTFactor import GraphTFactor

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
    elif n_layers == 33:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif n_layers == 36:
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif n_layers == 48:
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        model = model.half()
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
    path=f"models/{org}/GraphTF_{esm_model}.pth"
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net
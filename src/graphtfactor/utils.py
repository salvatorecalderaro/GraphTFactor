import esm 
from graphtfactor.model import GraphTFactor
import torch


def load_esm_model(device, n_layers):
    """
    Load an ESM model.

    Args:
        device (torch.device): The device to be used for PyTorch computations.
        n_layers (int): The number of layers in the ESM model.

    Returns:
        tuple[torch.nn.Module, esm.Alphabet]: A tuple containing the ESM model and its corresponding alphabet.

    Raises:
        ValueError: If the number of layers is not supported by the ESM model.
    """
    if n_layers == 6:
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        in_channesl=320
    elif n_layers == 12:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        in_channesl=480
    elif n_layers == 30:
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        in_channesl=640
    elif n_layers == 33:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        in_channesl=1280
    elif n_layers == 36:
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        in_channesl=2560
    elif n_layers == 48:
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        in_channesl=5120
    else:
        raise ValueError("Unsupported ESM model")
    
    model = model.to(device)
    model.eval()
    return model, alphabet,in_channesl


def load_model_from_file(org,esm_model,in_channels,device):
    """
    Load a PyTorch model from a file.

    Args:
        model_path (str): The path to the model file.
        device (torch.device): The device to be used for PyTorch computations.
    """
    
    net = GraphTFactor(in_channels=in_channels, dropout=0.2)
    net = net.to(device)
    path=f"checkpoints/{org}/GraphTF_{esm_model}.pth"
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net
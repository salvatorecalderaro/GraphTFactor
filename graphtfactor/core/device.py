import torch


def available_devices():

    devices = ["Automatic","CPU"]

    if torch.cuda.is_available():
        devices.append(f"CUDA - {torch.cuda.get_device_name(0)}")


    if (hasattr(torch.backends, "mps")and torch.backends.mps.is_available()):
        devices.append("MPS - Apple Silicon GPU")

    return devices
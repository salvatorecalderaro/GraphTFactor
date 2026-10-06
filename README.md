# GraphTFactor

GraphTFactor predicts whether a protein is a transcription factor using protein language model embeddings and a graph neural network. It segments protein sequences with a minimum description length method, builds a graph from the segments, and predicts a class for each sequence.

## Install

```bash
python -m pip install graphtfactor
```

Python 3.10 or newer is required. The package includes the pretrained GraphTFactor classifier checkpoints for all supported organism groups and ESM layer configurations. The first time a configuration is used, `fair-esm` downloads the corresponding ESM-2 model weights; internet access is needed for that download. ESM-2 models range from 8 million to 15 billion parameters, so memory and accelerator requirements vary substantially. Start with 6 layers if you are unsure.

PyTorch supports CPU, NVIDIA CUDA, and Apple Silicon MPS, subject to the PyTorch build installed for your platform. For CUDA, install the matching PyTorch build using the instructions on the [official PyTorch site](https://pytorch.org/get-started/locally/) before installing GraphTFactor.

## Quick start

```python
import torch
from graphtfactor import GraphTFactor

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = GraphTFactor(device=device, esm_layers=6, org="Virus")
predictions, probabilities = model.predict([
    "MDQYITLVELYIYDCNLFKSKNLKSFYKVHRVPEGDIVPKRRGGQLAGVTKSWVETNLVH"
])

print(predictions)    # class: 0 (non-TF) or 1 (TF)
print(probabilities)  # probability for the predicted class
```

`predict()` takes a list of amino-acid sequences and returns two lists, one with predicted class labels and one with their predicted-class probabilities. Unrecognized amino-acid symbols are treated as `X`.

## Supported models

Choose `org` from `"All"`, `"Virus"`, `"Eukaryotic"`, or `"Prokaryotic"`. Choose `esm_layers` from `6`, `12`, `30`, `33`, `36`, or `48`.

| `esm_layers` | ESM-2 model | Parameters |
| ---: | --- | ---: |
| 6 | `esm2_t6_8M_UR50D` | 8M |
| 12 | `esm2_t12_35M_UR50D` | 35M |
| 30 | `esm2_t30_150M_UR50D` | 150M |
| 33 | `esm2_t33_650M_UR50D` | 650M |
| 36 | `esm2_t36_3B_UR50D` | 3B |
| 48 | `esm2_t48_15B_UR50D` | 15B |

Larger ESM-2 models require considerably more memory and may not run on every device.

## Streamlit demo and source

The repository also contains a Streamlit demo and research experiments. See the [GitHub repository](https://github.com/salvatorecalderaro/GraphTFactor) for the application, source code, and updates.

## Citation

A paper citation will be added here when the publication details are available.

## License

GraphTFactor is distributed under the MIT License. See [LICENSE](LICENSE).

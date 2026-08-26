# GraphTFactor

<p align="center">
  <img src="images/logo.jpeg" alt="GraphTFactor Logo" width="280"/>
</p>

<p align="center">
  <strong>GraphTFactor: Modeling Protein Sequences as Graphs for Accurate Transcription Factor Prediction.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/">PyPI</a> •
  <a href="#streamlit-demo">Streamlit Demo</a> •
  <a href="#python-package">Python Package</a> •
  <a href="#citation">Citation</a>
</p>

---

## Introduction

**GraphTFactor** is a deep-learning framework for protein function prediction based on a graph representation of protein sequences.

Instead of processing a protein sequence only as a conventional string, GraphTFactor converts it into a graph representation. Protein sequences are segmented using a **Minimum Description Length** (MDL) approach, while **ESM embeddings** and sequence similarity are used to represent and characterize the resulting segments.

The resulting graph is processed by a **Graph Neural Network (GNN)** to learn structural representations of the protein and predict its functional class.

### How it works

The GraphTFactor pipeline can be summarized as:

<p align="center">
  <img src="images/pipeline.png" alt="GraphTFactor Pipeline"/>
</p>

More specifically:


1. **Protein sequence input**

   GraphTFactor accepts protein sequences in standard FASTA or string formats.

2. **Sequence segmentation**

   The sequence is segmented into informative subsequences using a data-driven segmentation strategy.

3. **ESM-2 embeddings**
    
    The sequen

3. **Graph construction**

   The resulting segments are used to construct a graph representation of the protein.

4. **Node representation**

   Each graph node is associated with a learned or pretrained protein representation.

5. **Graph Neural Network**

   The graph is processed using graph convolution/message-passing layers to capture relationships between sequence segments.

6. **Prediction**

   The learned graph representation is passed to a classifier to obtain the final protein function prediction.

This representation allows GraphTFactor to combine **sequence information, local protein representations, and graph topology** within a unified learning framework.

---

## Key Features

- 🧬 Protein sequence analysis from FASTA files
- 🕸️ De Bruijn graph representation of protein sequences
- 🧠 Graph Neural Network-based prediction
- 🔬 Support for pretrained protein embeddings
- ⚡ GPU acceleration with PyTorch
- 🖥️ Interactive Streamlit interface
- 📦 Installable Python package
- 🔎 Designed for research and large-scale protein analysis

---

# Streamlit Demo

GraphTFactor includes an interactive **Streamlit web application** that allows users to run predictions without writing Python code.

The interface provides a simple workflow:

```text
Upload FASTA
     │
     ▼
Select model
     │
     ▼
Select organism
     │
     ▼
Run GraphTFactor
     │
     ▼
Prediction
```

### Run the demo locally

After installing GraphTFactor, launch the Streamlit application with:

```bash
streamlit run app.py
```

The application allows users to:

- upload a FASTA file;
- select the protein embedding model;
- select the target organism/domain;
- choose the available computational device;
- run GraphTFactor inference;
- inspect the prediction results.

### Demo

<p align="center">
  <img src="docs/streamlit_demo.png" alt="GraphTFactor Streamlit Demo" width="850"/>
</p>

> **Try GraphTFactor interactively:**  
> Add the public Streamlit deployment URL here once available.

---

# Python Package

GraphTFactor is distributed as a Python package through **PyPI**, making it possible to integrate the model directly into existing bioinformatics and machine-learning workflows.

## Installation

Install the latest version with:

```bash
pip install graphtfactor
```

Verify the installation:

```python
import graphtfactor

print(graphtfactor.__version__)
```

---

## Quick Start

A minimal example is:

```python
import torch
from graphtfactor import GraphTFactor

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = GraphTFactor(
    device=device,
    esm_layers=30,
    org="virus"
)

prediction = model.predict("protein.fasta")

print(prediction)
```

GraphTFactor automatically selects the available computational backend:

```text
CUDA → NVIDIA GPU
MPS  → Apple Silicon GPU
CPU  → CPU fallback
```

---

## FASTA Example

GraphTFactor can be used directly with protein sequences stored in FASTA format:

```python
from graphtfactor import GraphTFactor

model = GraphTFactor(
    device="cuda",
    esm_layers=30,
    org="virus"
)

results = model.predict("Virus.fasta")

print(results)
```

For batch analysis, a FASTA file containing multiple protein sequences can be supplied to the same workflow.

---

# Citation

If you use **GraphTFactor** in your research, please cite:

```bibtex
@article{calderaro2026graphtfactor,
  title     = {GraphTFactor: Graph-based protein function prediction using De Bruijn graphs},
  author    = {Calderaro, Salvatore},
  year      = {2026},
  journal   = {TBD},
  doi       = {TBD}
}
```

If the associated paper is not yet published, please cite the GitHub repository:

```bibtex
@software{graphtfactor,
  author  = {Calderaro, Salvatore},
  title   = {GraphTFactor},
  year    = {2026},
  url     = {https://github.com/YOUR_USERNAME/GraphTFactor}
}
```

---

# License

GraphTFactor is released under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

# Acknowledgements

GraphTFactor builds upon open-source technologies and pretrained protein language models, including:

- PyTorch
- PyTorch Geometric
- ESM
- Biopython
- scikit-learn
- Streamlit

We acknowledge the developers and research groups behind these projects.

---

<p align="center">
  <strong>GraphTFactor</strong><br>
  Graph representations for protein function prediction.
</p>

import torch
import pandas as pd
from graphtfactor import GraphTFactor
from Bio import SeqIO
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# Label mapping
mapping = {
    "no-tf": 0,
    "tf": 1
}

fasta_path = "Virus.fasta"

seqs = []
true_labels = []
ids = []

# Read FASTA
with open(fasta_path, "r") as fasta_file:
    for record in tqdm(
        SeqIO.parse(fasta_file, "fasta"),
        desc="Reading FASTA",
        unit="sequence"
    ):
        info = record.description.split(" ")

        seq_id = info[0]
        label = info[1]

        seqs.append(str(record.seq))
        true_labels.append(mapping[label])
        ids.append(seq_id)

print(f"Total sequences read: {len(seqs)}")




# Load model
graphtf = GraphTFactor(
    device=device,
    esm_layers=6,
    org="Virus"
)

# Prediction
predictions, probas = graphtf.predict(seqs)


# Create readable labels
label_names = {
    0: "no-tf",
    1: "tf"
}

predicted_labels = [
    label_names[int(pred)] for pred in predictions
]

true_label_names = [
    label_names[int(label)] for label in true_labels
]

# Results table
results = pd.DataFrame({
    "ID": ids,
    "True": true_label_names,
    "Predicted": predicted_labels,
    "Probability": probas
})

results["Correct"] = results["True"] == results["Predicted"]

print("\n" + "=" * 70)
print("GraphTFactor Prediction Results")
print("=" * 70)

print(results.to_string(index=False))

print("=" * 70)


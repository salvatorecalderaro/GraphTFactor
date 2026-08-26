from sklearnex import patch_sklearn
patch_sklearn()
import argparse
import random 
import os 
import numpy as np 
import torch 
import platform
import cpuinfo
from datetime import datetime
from Bio import SeqIO
from tqdm import tqdm
import esm
from sklearn.model_selection import StratifiedKFold
from skorch import NeuralNetClassifier
from sklearn.metrics import recall_score,f1_score,balanced_accuracy_score,roc_auc_score,accuracy_score
import pandas as pd
import yaml
from time import perf_counter as pc

seed = 2026
N_FOLDS = 10
batch_size = 128


def parse_arguments():
    """
    Parse command-line arguments for dataset and model selection.
    Returns:
        tuple: A tuple containing the selected dataset and model."""
    data_choices = ["All","Eukaryotic","Prokaryotic","Virus"]
    models = [6,12,30,33,36,48]
    parser = argparse.ArgumentParser(description="Choose dataset")
    parser.add_argument("-d","--dataset", type=str, choices=data_choices, default="All",help="Dataset to use: All, Eukaryotic, Prokaryotic, Virus")
    parser.add_argument("-m","--model", type=int, choices=models, default=6,help="ESM Model to use: 6, 12, 30, 33, 36, 48")
    args = parser.parse_args()
    d, m = args.dataset, args.model
    return d, m

def identify_device():
    """
    Identify the available device for PyTorch computations (CPU, CUDA, or MPS).
    Returns:
        tuple: A tuple containing the identified device and its name."""
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    

def load_seqs(dataset):
    """
    Load sequences and labels from a FASTA file based on the specified dataset.

    Args:
        dataset (str): The name of the dataset.

    Returns:
        tuple: A tuple containing a list of sequences and a numpy array of labels.
    """
    path = f"../datasets/{dataset}.fasta"
    mapping = {"tf": 1, "no-tf": 0}
    sequences,labels=[],[]
    with open(path) as handle:
        for record in SeqIO.parse(handle,"fasta"):
            seq=str(record.seq)
            sequences.append(seq)
            label = str(record.description).split(" ")[-1]
            labels.append(mapping[label])
            
    dist = np.unique(labels, return_counts=True)
    class_dist = dict(zip(dist[0], dist[1]))
    print(f"Number of TF sequences {class_dist[1]}")
    print(f"Number of noTF sequences {class_dist[0]}")
    print(f"Number of sequences in the dataset {len(sequences)}")
    
    labels = np.array(labels)
    return sequences, labels

def load_model(n_layers, device):
    """
    Load an ESM model based on the specified number of layers.
    
    Args:
        n_layers (int): The number of layers in the ESM model.
        device (torch.device): The device to load the model onto.
    Returns:
        tuple: A tuple containing the loaded ESM model and its corresponding alphabet.
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

def compute_esm_embeddings(device,model,alphabet,m,sequences):
    """
    Compute ESM embeddings for a list of sequences using the specified ESM model.

    Args:
        device (torch.device): The device to perform computations on.
        model (torch.nn.Module): The ESM model.
        alphabet: The alphabet corresponding to the ESM model.
        m (int): The layer from which to extract embeddings.
        sequences (list): A list of protein sequences.

    Returns:
        np.ndarray: An array of protein embeddings.
    """
    model.eval()
    embeddings=[]
    batch_converter = alphabet.get_batch_converter()
    print("Computing ESM embeddings...")
    with torch.no_grad():
        for seq in tqdm(sequences):
            _, _, batch_tokens = batch_converter([("",seq)])
            batch_tokens = batch_tokens.to(device)
            results = model(batch_tokens, repr_layers=[m])
            token_embeddings = results["representations"][m]
            tokens_len = (batch_tokens != alphabet.padding_idx).sum(1)[0]
            residue_embeddings = token_embeddings[0, 1:tokens_len-1]
            protein_embedding = residue_embeddings.mean(dim=0).detach().cpu().numpy()
            embeddings.append(protein_embedding)
            
    embeddings=np.array(embeddings)
    
    print(f"Embedding dimensionality: {embeddings.shape}")
    return embeddings


def mlp_model(x_train, y_train, x_test, device):
    """
    Train a Multi-Layer Perceptron (MLP) model using the provided training data and evaluate it on the test data.

    Args:
        x_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test feature matrix.
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing predictions, predicted probabilities, and training time.
    """
    net = NeuralNetClassifier(

    module=torch.nn.Sequential(

        torch.nn.Linear(x_train.shape[1], 256),

        torch.nn.ReLU(),

        torch.nn.Linear(256, 128),

        torch.nn.ReLU(),

        torch.nn.Linear(128, 64),
        
        torch.nn.ReLU(),
        
        torch.nn.Linear(64, 2)

    ),

    device=device,

    max_epochs=50, 

    lr=0.001,

    batch_size=batch_size,

    optimizer=torch.optim.Adam,

    criterion=torch.nn.CrossEntropyLoss,
    train_split=None

    )
    s = pc()
    net.fit(x_train.astype(np.float32), y_train.astype(np.int64))
    e = pc()
    elapsed_time = (e - s) * 1000
    print(f"Training time: {elapsed_time:.2f} ms")

    preds = net.predict(x_test.astype(np.float32))
    proba = net.predict_proba(x_test.astype(np.float32))[:, 1]
    return preds, proba, elapsed_time

def evaluate_model(f,targets, preds, proba,t):
    """
    Evaluate the performance of a model using various metrics.
    
    Args:
        f (int): The fold number.
        targets (list): The true labels.
        preds (list): The predicted labels.
        proba (list): The predicted probabilities.
        t (float): The training time.
    Returns:
        list: A list containing the fold number, balanced accuracy, accuracy, sensitivity, specificity,              F1 score, ROC AUC, and training time.
    """
    bal_acc = balanced_accuracy_score(targets, preds)
    acc = accuracy_score(targets, preds)
    sensitivity = recall_score(targets, preds, pos_label=1)
    specificity = recall_score(targets, preds, pos_label=0)
    f1 = f1_score(targets, preds)
    roc_auc = roc_auc_score(targets, proba)
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    return [f,bal_acc,acc,sensitivity,specificity,f1,roc_auc,t]

def aggregate_results(results,dataset,m):
    """
    Aggregate the results of multiple folds and save them to a CSV file and a YAML file.
    
    Args:
        results (list): List of results from each fold.
        dataset (str): The name of the dataset.
        m (int): The number of layers in the ESM model.
    """
    
    RES_DIR = f"../experiments/Ablation/{dataset}"
    os.makedirs(RES_DIR, exist_ok=True)
    
    columns = ["Fold","Balanced Accuracy","Accuracy","Sensitivity","Specificity","F1 Score","ROC AUC","Training Time"]
    df = pd.DataFrame(results, columns=columns)
    path = f"../experiments/Ablation/{dataset}/mlp_{m}_metrics.csv"
    df.to_csv(path, index=False)

    results = {}
    print("\nAggregated Results:")
    for col in columns[1:]:
        results[col] = [float(df[col].mean()), float(df[col].std())]
        print(f"{col} - Mean: {results[col][0]:.4f} - Std: {results[col][1]:.4f}")
    
    path = f"../experiments/Ablation/{dataset}/mlp_{m}_results.yaml"
    with open(path, 'w') as f:
        yaml.dump(results, f)

def run_exp(device,model, dataset, esm_embeddings, labels):
    """
    Run the experiment using Stratified K-Fold cross-validation.

    Args:
        device (torch.device): The device to perform computations on.
        model (str): The model name.
        dataset (str): The dataset name.
        esm_embeddings (np.ndarray): The ESM embeddings.
        labels (np.ndarray): The labels.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    results = []
    width = 40 
    for fold, (train_index, test_index) in enumerate(skf.split(esm_embeddings, labels),start=1):
        print("\n" + f" Fold {fold} START ".center(width, "+"))
        x_train, x_test = esm_embeddings[train_index], esm_embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_preds, y_proba, t = mlp_model(x_train, y_train, x_test, device)
        fold_res =evaluate_model(fold, y_test, y_preds, y_proba, t)
        results.append(fold_res)
        print("\n" + f" Fold {fold} END ".center(width, "+"))
    aggregate_results(results, dataset, model)
    
def main():
    """
    Main function to run the MLP experiment.
    1. Parse command-line arguments for dataset and model selection.
    2. Identify the available device for PyTorch computations.
    3. Load sequences and labels from the specified dataset.
    4. Load the specified ESM model and compute embeddings for the sequences.
    5. Run the experiment using Stratified K-Fold cross-validation.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset, model = parse_arguments()
    device, dev_name = identify_device()
    print("----------------------------------------------------------------")
    print(" ESM2 Embeddings + MLP")
    print(f"Device: {device} - {dev_name}")
    print(f"Dataset: {dataset}")
    print(f"ESM model with  {model} layers")
    sequences, labels = load_seqs(dataset)
    esm_model, alphabet = load_model(model, device)
    esm_embeddings = compute_esm_embeddings(device, esm_model, alphabet, model, sequences)
    run_exp(device, model, dataset, esm_embeddings, labels)
    print("----------------------------------------------------------------")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
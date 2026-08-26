import argparse
import platform
import cpuinfo
import random
from Bio import SeqIO
import torch
from prot2graphrand import create_graph
from datetime import datetime
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import yaml
from GraphTFactor import GraphTFactor, train_net, predict
from sklearn.metrics import recall_score,f1_score,balanced_accuracy_score,roc_auc_score,accuracy_score
import pandas as pd


seed = 2026
NFOLDS = 10
batch_size = 64
epochs = 100
lr = 0.001
dropout = 0.2

def parse_arguments():
    """
    Parse command-line arguments for dataset and embedding size selection.
    Returns:
        tuple: A tuple containing the selected dataset and embedding size.
    """
    data_choices = ["All","Eukaryotic","Prokaryotic","Virus"]
    sizes = [320,480,640,1280,2560,5120]
    parser = argparse.ArgumentParser(description="Choose dataset")
    parser.add_argument("-d","--dataset", type=str, choices=data_choices, default="All",help="Dataset to use: All, Eukaryotic, Prokaryotic, Virus")
    parser.add_argument("-es","--enmbsize", type=int, choices=sizes, default=320,help="Emb size to use")
    args = parser.parse_args()
    d, m = args.dataset, args.enmbsize
    return d, m

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
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name


def set_seed(seed):
    """
    Set the random seed for reproducibility across various libraries and frameworks.
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    
def create_aa_embeddings(embedding_dim):
    """
    Create random embeddings for amino acids.
    Args:
        embedding_dim (int): The dimension of the embeddings.
    Returns:
        dict: A dictionary mapping amino acids to their random embeddings.
    """
    aminoacids = 'ACDEFGHIKLMNPQRSTVWYX'
    aa_embeddings = {}
    for aa in aminoacids:
        aa_embeddings[aa] = torch.randn(embedding_dim)
    return aa_embeddings


def create_graphs(dataset, aa_embeddings):
    """
    Create graphs from the sequences in the specified dataset using random amino acid embeddings.
    Args:
        dataset (str): The name of the dataset.
        aa_embeddings (dict): A dictionary mapping amino acids to their random embeddings.
    Returns:
        list: A list of graph objects created from the sequences in the dataset.
    """
    path = f"../datasets/{dataset}.fasta"
    mapping = {
        "no-tf": 0,
        "tf": 1
    }

    graphs = []
    with open(path) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        
        
    for record in tqdm(records, desc="Creating graphs", unit="graph"):
        sequence = str(record.seq)
        label_str = record.description.split(" ")[-1]
        label = mapping[label_str]
        g = create_graph(sequence, label, aa_embeddings, percentile=0.7, return_segments=False)
        graphs.append(g)
    
    return graphs


def create_loaders(graphs, train_index, test_index):
    """
    Create DataLoader objects for training and testing datasets.
    Args:
        graphs (list): A list of graph objects.
        train_index (list): Indices for the training dataset.
        test_index (list): Indices for the testing dataset.
    Returns:
        tuple: A tuple containing the training and testing DataLoader objects.
    """
    train_graphs = [graphs[i] for i in train_index]
    test_graphs = [graphs[i] for i in test_index]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    print(f"Train graphs: {len(train_graphs)} - Test graphs: {len(test_graphs)}")
    return train_loader, test_loader

def evaluate_model(f,targets, preds, proba,t):
    """
    Evaluate the model's performance using various metrics.
    Args:
        f (int): The fold number.
        targets (list): The true labels of the test dataset.
        preds (list): The predicted labels from the model.
        proba (list): The predicted probabilities from the model.
        t (float): The training time for the model.
    Returns:
        list: A list containing the fold number, balanced accuracy, accuracy, sensitivity, specificity, f1 score, ROC AUC, and training time.
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


def aggregate_results(results,dataset,emb_size):
    """
    Aggregate the evaluation results and save them to CSV and YAML files.
    Args:
        results (list): A list of evaluation results for each fold.
        dataset (str): The name of the dataset.
        emb_size (int): The size of the random embeddings.
    """
    columns = ["Fold","Balanced Accuracy","Accuracy","Sensitivity","Specificity","F1 Score","ROC AUC","Training Time"]
    df = pd.DataFrame(results, columns=columns)
    path = f"../experiments/Ablation/{dataset}/rand_emb_{emb_size}_metrics.csv"
    df.to_csv(path, index=False)

    results = {}
    print(f"\nAggregated Results")
    for col in columns[1:]:
        results[col] = [float(df[col].mean()), float(df[col].std())]
        print(f"{col} - Mean: {results[col][0]:.4f} - Std: {results[col][1]:.4f}")
    
    path = f"../experiments//Ablation/{dataset}/rand_emb_{emb_size}_results.yaml"
    with open(path, 'w') as f:
        yaml.dump(results, f)
    

def run_exp(device, graphs,dataset, emb_size):
    """
    Run the experiment using stratified k-fold cross-validation.
    Args:
        device (torch.device): The device to be used for PyTorch computations.
        graphs (list): A list of graph objects.
        dataset (str): The name of the dataset.
        emb_size (int): The size of the random embeddings.
    """
    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    labels = [g.y.item() for g in graphs]
    results = []
    width = 40 
    for fold, (train_index, test_index) in enumerate(skf.split(graphs, labels),start=1):
        print("\n" + f" Fold {fold} START ".center(width, "+"))
        train_loader, test_loader = create_loaders(graphs, train_index, test_index)
        model = GraphTFactor(in_channels=graphs[0].num_node_features, dropout=dropout).to(device)
        trained_model,t = train_net(device, model, train_loader, epochs, lr)
        targets, preds, proba = predict(device, trained_model, test_loader)
        fold_results = evaluate_model(fold,targets, preds, proba,t)
        results.append(fold_results)
        print("\n" + f" Fold {fold} END ".center(width, "+"))
    aggregate_results(results,dataset,emb_size)
    print("Experiment completed successfully.")
    return      

def main():
    """
    Main function to run the GraphTFactor experiment with random embeddings.
    1. Print the current date and time.
    2. Parse command-line arguments to select the dataset and embedding size.
    3. Identify the available device for PyTorch computations.
    4. Create random embeddings for amino acids.
    5. Create graphs for the selected dataset using the random embeddings.
    6. Run the experiment using stratified k-fold cross-validation.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset, emb_size = parse_arguments()
    device, dev_name = identify_device()
    print("----------------------------------------------------------------")
    print("Graph TFactor with Random Embeddings")
    print(f"Device: {device} - {dev_name}")
    print(f"Dataset: {dataset}")
    print(f"Random Embedding Size: {emb_size}")
    aa_embeddings = create_aa_embeddings(emb_size)
    graphs = create_graphs(dataset, aa_embeddings)
    run_exp(device, graphs,dataset, emb_size)
    print("----------------------------------------------------------------\n")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return 

if __name__ == "__main__":
    main()
import random
import argparse
import os 
import numpy as np 
import torch 
import platform
import cpuinfo
from Bio import SeqIO
import gc
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from DeepTFactor import DeepTFactor,train_net,predict
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score,balanced_accuracy_score
import pandas as pd
import yaml
from datetime import datetime

epochs=50
nfolds=10
mini_batch_size=128
lr=0.001
seed=2026

def set_seed(seed):
    """
    Set the random seed for reproducibility.
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


def parse_arguments():
    """
    Parse command-line arguments to select the dataset.
    Returns:
        str: The selected dataset.
    """
    parser = argparse.ArgumentParser(description="Choose dataset")
    choices = ["All","Eukaryotic","Prokaryotic","Virus"]
    parser.add_argument("-d","--dataset", type=str, choices=choices, default="All",help="Dataset to use: All, Eukaryotic, Prokaryotic, Virus")
    args = parser.parse_args()
    return args.dataset


def identify_device():
    """
    Identify the available device (CPU, GPU, or MPS) for computation.
    Returns:
        tuple: A tuple containing the device and its name.
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

def get_aa_map():
    """
    Get the amino acid one-hot encoding map.
    Returns:
        dict: A dictionary mapping amino acids to one-hot encoded vectors.
    """
    aa_vocab = ['A', 'C', 'D', 'E', 
                'F', 'G', 'H', 'I', 
                'K', 'L', 'M', 'N', 
                'P', 'Q', 'R', 'S',
                'T', 'V', 'W', 'X', 
                'Y', '_']
    map = {}
    for i, char in enumerate(aa_vocab):
        baseArray = np.zeros(len(aa_vocab) - 1)
        if char != '_':
            baseArray[i] = 1
        map[char] = baseArray
    return map

def pad_sequence(sequence):
    """ 
    Pad the input sequence with underscores to a maximum length of 1000.
    Args:
        sequence (str): The input amino acid sequence.
    Returns:
        str: The padded sequence.
    """
    max_length = 1000
    if len(sequence) < max_length:
        sequence += '_' * (max_length - len(sequence))
    return sequence

def one_hot(sequence):
    """
    Convert an amino acid sequence into a one-hot encoded representation.
    Args:
        sequence (str): The input amino acid sequence.
    Returns:
        np.ndarray: A 2D array representing the one-hot encoded sequence.
    """
    map = get_aa_map()
    single_onehot = []
    if len(sequence) < 1000:
        sequence = pad_sequence(sequence)

    for x in sequence:
        single_onehot.append(map[x])
    return np.asarray(single_onehot)

def load_and_encode_sequences(dataseet):
    """
    Load and encode amino acid sequences from a FASTA file.
    Args:
        dataseet (str): The name of the dataset to load.
    Returns:
        tuple: A tuple containing the encoded sequences and their corresponding labels.
    """
    sequences, labels = [], []
    mapping = {"tf": 1, "no-tf": 0}
    path = f"../datasets/{dataseet}.fasta"
    print(f"Encoding sequences......")
    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            feat = one_hot(sequence)
            sequences.append(feat)
            label = str(record.description).split(" ")[-1]
            labels.append(mapping[label])

    sequences = np.asarray(sequences)
    labels = np.array(labels).reshape(-1)
    print("Number of sequences in the dataset:", sequences.shape[0])
    print("Data dimensionality", sequences.shape[1], sequences.shape[2])
    return sequences, labels

def create_train_test_loaders(sequences,labels,train,test):
    """
    Create DataLoader objects for training and testing datasets.
    Args:
        sequences (np.ndarray): The encoded sequences.
        labels (np.ndarray): The corresponding labels.
        train (list): Indices for the training set.
        test (list): Indices for the test set.
    Returns:
        tuple: A tuple containing the training and testing DataLoader objects.
    """
    x = sequences[train]
    y = labels[train]

    print("Number of sequences in the training set:", len(x))
    data = list(zip(x, y))
    trainloader = DataLoader(data, shuffle=True, batch_size=mini_batch_size)

    x = sequences[test]
    y = labels[test]

    print("Number of sequences in the test set:", len(x))
    data = list(zip(x, y))
    testloader = DataLoader(data, shuffle=False, batch_size=mini_batch_size)

    return trainloader, testloader

def evaluate_model(f,targets, preds, proba):
    """
    Evaluate the model's performance using various metrics.
    Args:
        f (int): The fold number.
        targets (np.ndarray): The true labels.
        preds (np.ndarray): The predicted labels.
        proba (np.ndarray): The predicted probabilities.
    Returns:
        list: A list containing the fold number and calculated metrics.
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
    return [f,bal_acc,acc,sensitivity,specificity,f1,roc_auc]

def save_report(device,results,dataset):
    """
    Save the evaluation metrics to a CSV file and a YAML file.
    Args:
        device (str): The name of the device used for computation.
        results (list): A list of evaluation metrics for each fold.
        dataset (str): The name of the dataset.
    """
    
    columns = ["Fold","Balanced Accuracy","Accuracy","Sensitivity","Specificity","F1 Score","ROC AUC","Training Time"]  
    metrics = pd.DataFrame(results, columns=columns)
    
    folder=f"../experiments/DeepTFactor/{dataset}"
    
    os.makedirs(folder, exist_ok=True)
    
    metrics_path = f"../experiments/DeepTFactor/{dataset}/metrics.csv"
    
        
    metrics.to_csv(metrics_path, index=False)

    columns = ["Fold","Balanced Accuracy","Accuracy","Sensitivity","Specificity","F1 Score","ROC AUC","Training Time"]

    data = {}
    print(f"Average results:")
    for column in metrics.columns[1:]:
        values = metrics[column]
        mu, sigma = np.mean(values), np.std(values)
        print(f"{column}: mean {mu} sd {sigma}")
        data[column] = {
            "Mean": float(mu),
            "Standard Deviation": float(sigma)
        }

    res_path = f"../experiments/DeepTFactor/{dataset}/results.yaml"
    data["Device"]=device
    with open(res_path, "w") as file:
        yaml.dump(data, file)
        
def print_gpu_memory_usage():
    """
    Print the current GPU memory usage.
    """
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Allocated memory: {allocated / 1e6} MB")
    print(f"Reserved memory: {reserved / 1e6} MB")
    
    
def free_gpu_memory(net,trainloader,testloader):
    """
    Free the GPU memory occupied by the network and data loaders.
    Args:
        net (torch.nn.Module): The neural network model.
        trainloader (DataLoader): The training data loader.
        testloader (DataLoader): The testing data loader.
    """
    print_gpu_memory_usage()
    print("Freeing GPU memory...")
    del trainloader, testloader
    del net 
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print("GPU memory freed.")
    print_gpu_memory_usage()

def run_experiment(device,devname,dataset):
    """
    Run the DeepTFactor experiment using stratified k-fold cross-validation.
    Args:
        device (torch.device): The device to use for computation.
        devname (str): The name of the device.
        dataset (str): The name of the dataset to use.
    """
    sequences,labels=load_and_encode_sequences(dataset)
    skf=StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=seed)
    report=[]
    for fold, (train_index, test_index) in enumerate(skf.split(sequences, labels),start=1):
        print("===================================================")
        print(f"Fold {fold}/{nfolds}")
        trainloader,testloader=create_train_test_loaders(sequences,labels,train_index,test_index)
        net = DeepTFactor(out_features=[1]).to(device)
        net, t = train_net(device, net, trainloader, epochs, lr)
        y_true, y_pred, proba = predict(device,net, testloader)
        rep=evaluate_model(fold,y_true,y_pred,proba)
        rep.append(t)
        report.append(rep)
        free_gpu_memory(net,trainloader,testloader)
        print("===================================================")
    
    save_report(devname,report,dataset)
    print("Experiment completed successfully.")
        

def main():
    """
    Main function to run the DeepTFactor experiment.
    1. Identify the available device (CPU, GPU, or MPS).
    2. Parse command-line arguments to select the dataset.
    3. Load and encode the sequences from the selected dataset.
    4. Run the experiment using stratified k-fold cross-validation.
    5. Save the evaluation metrics to CSV and YAML files.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    device,devname=identify_device()
    print("--------Classify TF proteins using DeepTFactor--------")
    print(f"Using {device} - {devname}")
    dataset = parse_arguments()
    print(f"Dataset selected: {dataset}")
    print("----------------------------------------------------\n")
    run_experiment(device,devname,dataset)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exit(0)
    

if __name__=="__main__":
    main()
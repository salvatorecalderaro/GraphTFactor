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

seed=0
epochs=50
nfolds=10
mini_batch_size=128
lr=0.001


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set for random number generation. 
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
    Parse command-line arguments.
    Returns:
        str: The chosen dataset.
    """
    parser = argparse.ArgumentParser(description="Choose dataset")
    choices = ["All","Eukaryotic","Prokaryotic","Virus"]
    parser.add_argument("-d","--dataset", type=str, choices=choices, default="All",help="Dataset to use: All, Eukaryotic, Prokaryotic, Virus")
    args = parser.parse_args()
    return args.dataset


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

def get_aa_map():
    """
    Get the amino acid one-hot encoding map.
    Returns:
        dict: A dictionary mapping amino acid characters to one-hot encoded numpy arrays.
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
    Pad the input protein sequence to a maximum length of 1000 characters with underscores ('_').
    Args:
        sequence (str): The input protein sequence.
    Returns:
        str: The padded protein sequence.
    """
    max_length = 1000
    if len(sequence) < max_length:
        sequence += '_' * (max_length - len(sequence))
    return sequence

def one_hot(sequence):
    """
    Convert the input protein sequence into a one-hot encoded numpy array.
    Args:
        sequence (str): The input protein sequence.
    Returns:
        np.ndarray: A 2D numpy array representing the one-hot encoded sequence.
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
    Load and encode protein sequences from a FASTA file.
    Args:
        dataseet (str): The name of the dataset to load.
    Returns:
        tuple: A tuple containing the encoded sequences and their corresponding labels.
    """
    sequences, labels = [], []
    mapping = {"tf": 1, "no-tf": 0}
    path = f"../datasets/{dataseet}_el.fasta"
    
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
        sequences (np.ndarray): The encoded protein sequences.
        labels (np.ndarray): The corresponding labels for the sequences.
        train (list): List of indices for the training set.
        test (list): List of indices for the testing set.
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
        targets (list): The true labels.
        preds (list): The predicted labels.
        proba (list): The predicted probabilities.
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
    Save the evaluation report to a CSV file and a YAML file.
    Args:
        device (str): The device used for computation.
        results (list): List of results from each fold.
        dataset (str): The name of the dataset.
    """
    columns = ["Fold","Balanced Accuracy","Accuracy","Sensitivity","Specificity","F1 Score","ROC AUC","Training Time"]  
    metrics = pd.DataFrame(results, columns=columns)
    
    folder=f"../experiments/DeepTFactor/{dataset}"
    
    os.makedirs(folder, exist_ok=True)
    
    metrics_path = f"../experiments/DeepTFactor/{dataset}/metrics_el.csv"
    
        
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

    res_path = f"../experiments/DeepTFactor/{dataset}/results_el.yaml"
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
    Free GPU memory by deleting the model and data loaders, and clearing the cache.
    Args:
        net (torch.nn.Module): The trained model.
        trainloader (torch.utils.data.DataLoader): The training data loader.
        testloader (torch.utils.data.DataLoader): The testing data loader.
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
    Run the DeepTFactor experiment using Stratified K-Fold cross-validation.
    Args:
        device (torch.device): The device to be used for PyTorch computations.
        devname (str): The name of the device.
        dataset (str): The name of the dataset.
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
    1. Parse command-line arguments for dataset selection.
    2. Identify the available device for PyTorch computations.
    3. Load and encode protein sequences from the specified dataset.
    4. Run the experiment using Stratified K-Fold cross-validation.
    5. Print the results and save them to CSV and YAML files.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    device,devname=identify_device()
    print("--------Classify TF proteins using DeepTFactor--------")
    print(f"Using {device} - {devname}")
    dataset = parse_arguments()
    print(f"Dataset selected: {dataset} (equal length)")
    print("----------------------------------------------------\n")
    run_experiment(device,devname,dataset)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exit(0)
    

if __name__=="__main__":
    main()
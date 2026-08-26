import random
import argparse
import os 
import numpy as np 
import torch 
import platform
import cpuinfo
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from DeepTFactor import DeepTFactor,train_net,predict
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score,balanced_accuracy_score
from datetime import datetime

seed=2026
epochs=50
test_size = 0.1
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


def create_loaders(sequences, labels):
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=seed, stratify=labels)
    train_dataset = list(zip(X_train, y_train))
    test_dataset = list(zip(X_test, y_test))
    train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_model(targets, preds, proba):
    """
    Evaluate the performance of the logistic regression model using various metrics.
    Args:
        targets (np.ndarray): True labels for the test data.
        preds (np.ndarray): Predicted labels for the test data.
        proba (np.ndarray): Predicted probabilities for the positive class.
    Returns:
        None
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
    
    
def main():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset = parse_arguments()
    device, dev_name = identify_device()
    print("----------------------------------------------------------------")
    print("DeepTFactor - Model Training and Weights Saving")
    print("----------------------------------------------------------------")
    print(f"Device: {device} - {dev_name}")
    print(f"Dataset: {dataset}")
    print(f"Using {100-test_size*100:.2f}% of the data for training and {test_size*100:.2f}% for testing")
    sequences,labels=load_and_encode_sequences(dataset)
    train_loader, test_loader = create_loaders(sequences, labels)
    net = DeepTFactor(out_features=[1]).to(device)
    trained_model, t = train_net(device, net, train_loader, epochs, lr)
    targets, preds, proba = predict(device, trained_model, test_loader)
    evaluate_model(targets, preds, proba)
    folder = f"../models/{dataset}/"
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/DeepTFactor.pth"
    torch.save(trained_model.state_dict(), path)
    print("----------------------------------------------------------------\n")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return 

if __name__ == "__main__":
    main()
    
    
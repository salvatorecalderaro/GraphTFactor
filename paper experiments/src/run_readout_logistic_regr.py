from sklearnex import patch_sklearn
patch_sklearn()
import argparse
import numpy as np 
import torch 
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,f1_score,balanced_accuracy_score,roc_auc_score,accuracy_score
import pandas as pd
import yaml
from time import perf_counter as pc
from sklearn.preprocessing import StandardScaler

seed = 2026
N_FOLDS = 10

def parse_arguments():
    """
    Parse command line arguments for dataset and model selection.
    Returns:
        tuple: A tuple containing the selected dataset and model.
    """
    data_choices = ["All","Eukaryotic","Prokaryotic","Virus"]
    models = [6,12,30,33,36,48]
    parser = argparse.ArgumentParser(description="Choose dataset")
    parser.add_argument("-d","--dataset", type=str, choices=data_choices, default="All",help="Dataset to use: All, Eukaryotic, Prokaryotic, Virus")
    parser.add_argument("-m","--model", type=int, choices=models, default=6,help="ESM Model to use: 6, 12, 30, 33, 36, 48")
    args = parser.parse_args()
    d, m = args.dataset, args.model
    return d, m

def load_graphs(dataset, m):
    """
    Load graphs for the specified dataset and model.
    Args:
        dataset (str): The name of the dataset.
        m (int): The number of layers in the ESM model.
    Returns:
        list: A list of loaded graphs.
    """
    path = f"../graphs/{dataset}_{m}.pth"
    graphs = torch.load(path,weights_only=False)
    print(f"Number of graphs loaded: {len(graphs)}")
    return graphs


def apply_readout(graphs):
    """
    Apply readout operation on the graphs to extract features and labels.
    Args:
        graphs (list): A list of graphs.
    Returns:
        tuple: A tuple containing the extracted features and labels.
    """
    features = []
    labels = []
    for graph in tqdm(graphs, desc="Applying readout"):
        features.append(graph.x.mean(dim=0).numpy())
        labels.append(graph.y.item())
    print(f"Features shape: {np.array(features).shape}")
    
    ss = StandardScaler()
    features = ss.fit_transform(np.array(features))
    return features, np.array(labels)


def logistic_regression_model(x_train, y_train, x_test):
    """
    Train a logistic regression model on the training data and make predictions on the test data.
    Args:
        x_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test feature data.
    Returns:
        tuple: A tuple containing the predicted labels, predicted probabilities, and elapsed training time in milliseconds.
    """
    model = LogisticRegression(max_iter=1000, random_state=seed)
    s = pc()
    model.fit(x_train, y_train)
    e = pc()
    elapsed_time = (e - s)*1000
    print(f"Training time: {elapsed_time:.2f} ms")
    preds = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    return preds, proba, elapsed_time

def evaluate_model(f,targets, preds, proba,t):
    """
    Evaluate the performance of the logistic regression model using various metrics.
    Args:
        f (int): The fold number.
        targets (np.ndarray): True labels for the test data.
        preds (np.ndarray): Predicted labels from the model.
        proba (np.ndarray): Predicted probabilities from the model.
        t (float): Elapsed training time in milliseconds.
    Returns:
        list: A list containing the fold number, balanced accuracy, accuracy, sensitivity, specificity, F1 score, ROC AUC, and training time.
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
    Aggregate the results from all folds and save them to a CSV file and a YAML file.
    Args:
        results (list): A list of results from each fold.
        dataset (str): The name of the dataset used.
        m (int): The number of layers in the ESM model.
    """
    columns = ["Fold","Balanced Accuracy","Accuracy","Sensitivity","Specificity","F1 Score","ROC AUC","Training Time"]
    df = pd.DataFrame(results, columns=columns)
    path = f"../experiments/Ablation/{dataset}/log_regr_readout_{m}_metrics.csv"
    df.to_csv(path, index=False)

    results = {}
    print("\nAggregated Results:")
    for col in columns[1:]:
        results[col] = [float(df[col].mean()), float(df[col].std())]
        print(f"{col} - Mean: {results[col][0]:.4f} - Std: {results[col][1]:.4f}")
    
    path = f"../experiments/Ablation/{dataset}/log_regr_readout_{m}_results.yaml"
    with open(path, 'w') as f:
        yaml.dump(results, f)

def run_exp(model, dataset, esm_embeddings, labels):
    """
    Run the logistic regression experiment using stratified k-fold cross-validation.
    Args:
        model (int): The number of layers in the ESM model.
        dataset (str): The name of the dataset used.
        esm_embeddings (np.ndarray): The computed ESM embeddings for the sequences.
        labels (np.ndarray): The true labels for the sequences.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    results = []
    width = 40 
    for fold, (train_index, test_index) in enumerate(skf.split(esm_embeddings, labels),start=1):
        print("\n" + f" Fold {fold} START ".center(width, "+"))
        x_train, x_test = esm_embeddings[train_index], esm_embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        y_preds, y_proba, t = logistic_regression_model(x_train, y_train, x_test)
        fold_res =evaluate_model(fold, y_test, y_preds, y_proba, t)
        results.append(fold_res)
        print("\n" + f" Fold {fold} END ".center(width, "+"))
    aggregate_results(results, dataset, model)  
    
    
def main():
    """
    Main function to run the logistic regression experiment with ESM embeddings.
    1. Parse command line arguments for dataset and model selection.
    2. Load graphs for the specified dataset and model.
    3. Apply readout operation to extract features and labels from the graphs.
    4. Run the logistic regression experiment using stratified k-fold cross-validation.
    """
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    dataset, model = parse_arguments()
    print("----------------------------------------------------------------")
    print(" Graph Readouts  + logistic regression")
    print(f"Dataset: {dataset}")
    print(f"Graph built with ESM model with  {model} layers")
    graphs = load_graphs(dataset, model)
    embeddings, labels = apply_readout(graphs)
    run_exp(model, dataset, embeddings, labels)
    print("----------------------------------------------------------------")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
    
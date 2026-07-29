import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.profile import timeit
from tqdm import tqdm


class GraphTFactor(nn.Module):
    """
    GraphTFactor model using GraphSAGE layers and an MLP classifier.
    """
    def __init__(self,in_channels,dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(in_channels, 256),
            SAGEConv(256, 128),
            SAGEConv(128, 64),
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(256),
            nn.BatchNorm1d(128),
            nn.BatchNorm1d(64),
        ])

        # ===== MLP classifier =====
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.dropout = dropout

    def forward(self, data):
        """
        Forward pass of the GraphTFactor model.

        Parameters:
        - data (torch_geometric.data.Data): The input graph data.

        Returns:
        - torch.Tensor: The output logits of the model.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        
        x = torch.cat([x_max, x_mean], dim=1)  #
        return self.mlp(x).view(-1) 



def train_net(device, net, trainloader, epochs, lr):
    """
    Train the GraphTFactor model.

    Parameters:
    - device (torch.device): The device to run the training on.
    - net (GraphTFactor): The GraphTFactor model to be trained.
    - trainloader (torch_geometric.data.DataLoader): The DataLoader for the training data.
    - epochs (int): The number of training epochs.
    - lr (float): The learning rate for the optimizer.

    Returns:
    - net (GraphTFactor): The trained GraphTFactor model.
    - float: The training duration in seconds.
    """
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() 

    net.train()
    tc = timeit()

    with tc:
        for epoch in tqdm(range(epochs), desc="Training"):
            total_loss = 0.0

            for data in trainloader:
                data = data.to(device)
                optimizer.zero_grad()
                logits = net(data)
                loss = criterion(logits, data.y.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * data.num_graphs

            avg_loss = total_loss / len(trainloader.dataset)
            # print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")

    print(f"Training time: {tc.duration:.2f} seconds")
    return net, tc.duration


# ----------------- Prediction -----------------
def predict(device, net, dataloader, threshold=0.5):
    """
    Predict the output of the GraphTFactor model.

    Parameters:
    - device (torch.device): The device to run the prediction on.
    - net (GraphTFactor): The GraphTFactor model to be used for prediction.
    - dataloader (torch_geometric.data.DataLoader): The DataLoader for the prediction data.
    - threshold (float): The threshold for converting probabilities to binary predictions.

    Returns:
    - all_targets (list): The true labels of the data.
    - all_preds (list): The predicted labels of the data.
    - all_proba (list): The predicted probabilities of the data.
    """
    net.to(device)
    net.eval()

    all_targets = []
    all_preds = []
    all_proba = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Predicting"):
            data = data.to(device)

            logits = net(data)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()

            all_targets.extend(data.y.view(-1).cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_proba.extend(probs.cpu().numpy())

    return all_targets, all_preds, all_proba


def predict_graph(model,graph,device):
    null_batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    graph.batch = null_batch
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        logits = model(graph)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
    return preds.item(),probs.item()
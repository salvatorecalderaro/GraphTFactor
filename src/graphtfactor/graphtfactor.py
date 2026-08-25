from graphtfactor.utils import load_esm_model
from graphtfactor.utils import load_model_from_file
from graphtfactor.graph import create_graph
from graphtfactor.model import predict_graph
from tqdm import tqdm

class GraphTFactor:
    """
    GraphTFactor class for predicting protein function using graph neural networks and ESM embeddings.
    """
    def __init__(self,device,esm_layers,org):
        """
        Initializes the GraphTFactor class.

        Args:
            device (str): The device to run the models on.
            esm_layers (int): The number of ESM layers to use.
            org (str): The organism type.
        """
        self.device = device
        self.org = org
        self.esm_layers = esm_layers
        esm_model, alphabet,in_channels = load_esm_model(self.device, self.esm_layers)
        
        self.esm_model = esm_model
        self.alphabet = alphabet
        self.in_channels = in_channels
        
        self.gnn_model = load_model_from_file(self.org,self.esm_layers,self.in_channels,self.device)
        
    
    
    def predict(self,seqs):
        """
        Predicts the function of protein sequences.

        Args:
            seqs (list of str): A list of protein sequences.

        Returns:
            list of tuples: A list of predictions and probabilities for each sequence.
        """
        graphs = []
        for seq in seqs:
            graph = create_graph(seq, self.esm_model, self.alphabet, self.esm_layers, self.device, percentile=0.7)
            graphs.append(graph)
        
        predictions = []
        for graph in tqdm(graphs, desc="Predicting", unit="graph"):
            pred,prob = predict_graph(self.gnn_model,graph,self.device)
            predictions.append((pred,prob))
        return predictions
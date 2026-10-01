"""Run PyTorch/ESM inference outside Streamlit's script thread."""

import json
import sys

import torch

from graph import create_graph
from model import predict_graph
from utils import identify_device, load_model, load_model_from_file


def main():
    sequence, esm_model_arg, organism = sys.argv[1:4]
    esm_model = int(esm_model_arg)
    device, _ = identify_device()

    esm, alphabet = load_model(esm_model, device)
    graph = create_graph(
        sequence,
        esm,
        alphabet,
        esm_model,
        device=device,
        y=0,
    )

    classifier = load_model_from_file(
        organism,
        esm_model,
        graph.num_node_features,
        device,
    )
    prediction, probability = predict_graph(classifier, graph, device)

    edge_index = graph.edge_index.detach().cpu().tolist()
    result = {
        "prediction": int(prediction),
        "probability": float(probability),
        "graph": {
            "nodes": int(graph.num_nodes),
            "edge_count": int(graph.num_edges),
            "features": int(graph.num_node_features),
            "edges": list(map(list, zip(edge_index[0], edge_index[1]))),
        },
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

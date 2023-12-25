import os
import tarfile

import mlx.core as mx
import numpy as np
import requests
import scipy.sparse as sparse

"""
Preprocessing follows the same implementation as in:
https://github.com/tkipf/gcn
https://github.com/senadkurtisi/pytorch-GCN/tree/main
"""


def download_cora():
    """Downloads the cora dataset into a local cora folder."""

    url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    extract_to = "."

    if os.path.exists(os.path.join(extract_to, "cora")):
        return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(extract_to, url.split("/")[-1])

        # Write the file to local disk
        with open(file_path, "wb") as file:
            file.write(response.raw.read())

        # Extract the .tgz file
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"Cora dataset extracted to {extract_to}")

        os.remove(file_path)


def train_val_test_mask():
    """Splits the loaded dataset into train/validation/test sets."""

    train_set = mx.arange(140)
    validation_set = mx.arange(200, 500)
    test_set = mx.arange(500, 1500)

    return train_set, validation_set, test_set


def enumerate_labels(labels):
    """Converts the labels from the original
    string form to the integer [0:MaxLabels-1]
    """
    label_map = {v: e for e, v in enumerate(set(labels))}
    labels = np.array([label_map[label] for label in labels])
    return labels


def normalize_adjacency(adj):
    """Normalizes the adjacency matrix according to the
    paper by Kipf et al.
    https://arxiv.org/abs/1609.02907
    """
    adj = adj + sparse.eye(adj.shape[0])

    node_degrees = np.array(adj.sum(1))
    node_degrees = np.power(node_degrees, -0.5).flatten()
    node_degrees[np.isinf(node_degrees)] = 0.0
    node_degrees[np.isnan(node_degrees)] = 0.0
    degree_matrix = sparse.diags(node_degrees, dtype=np.float32)

    adj = degree_matrix @ adj @ degree_matrix
    return adj


def load_data(config):
    """Loads the Cora graph data into MLX array format."""
    print("Loading Cora dataset...")

    # Download dataset files
    download_cora()

    # Graph nodes
    raw_nodes_data = np.genfromtxt(config.nodes_path, dtype="str")
    raw_node_ids = raw_nodes_data[:, 0].astype(
        "int32"
    )  # unique identifier of each node
    raw_node_labels = raw_nodes_data[:, -1]
    labels_enumerated = enumerate_labels(raw_node_labels)  # target labels as integers
    node_features = sparse.csr_matrix(raw_nodes_data[:, 1:-1], dtype="float32")

    # Edges
    ids_ordered = {raw_id: order for order, raw_id in enumerate(raw_node_ids)}
    raw_edges_data = np.genfromtxt(config.edges_path, dtype="int32")
    edges_ordered = np.array(
        list(map(ids_ordered.get, raw_edges_data.flatten())), dtype="int32"
    ).reshape(raw_edges_data.shape)

    # Adjacency matrix
    adj = sparse.coo_matrix(
        (np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
        shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
        dtype=np.float32,
    )

    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj)
    adj = normalize_adjacency(adj)

    # Convert to mlx array
    features = mx.array(node_features.toarray(), mx.float32)
    labels = mx.array(labels_enumerated, mx.int32)
    adj = mx.array(adj.toarray())

    print("Dataset loaded.")
    return features, labels, adj

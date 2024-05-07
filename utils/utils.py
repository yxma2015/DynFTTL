"""
Author: Ma Yaxiong
E-mail: mayaxiong@xidian.edu.cn
Description:  to be added
Time: 2024/1/21
Filename: utils.py
"""
import torch
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import SpectralClustering


def relu(x):
    return torch.nn.functional.relu(torch.from_numpy(x)).numpy() + 1e-5


def get_laplace_matrix(x):
    degree_matrix = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        degree_matrix[i, i] = sum(x[i, :])
    return degree_matrix - x


def nmf(x: np.array, dimension: np.array) -> np.array:
    """do NMF(non-negative matrix factorization) with a given matrix x and expected dimension.

    Args:
        x (np.array): non-negative matrix X to be factorized
        dimension (np.array): dimension

    Returns:
        np.array: (W, H) whose product approximates the non-negative matrix X
    """
    model = NMF(n_components=dimension, init='random', random_state=0, max_iter=400)
    w = model.fit_transform(x)
    h = model.components_
    return w, h


def compute_pmi_matrix(adjacency_matrix: np.array, k: int) -> np.array:
    """_summary_

    Args:
        adjacency_matrix (np.array): _description_
        k(np.array): constant integer

    Returns:
        np.array: pmi matrix
    """
    degree_nodewise = np.sum(adjacency_matrix, axis=0)
    num_node = np.sum(degree_nodewise)
    row, col = np.nonzero(adjacency_matrix)
    pmi_matrix = adjacency_matrix
    for i in range(len(row)):
        score = np.log(
            adjacency_matrix[row[i], col[i]] *
            num_node / (degree_nodewise[row[i]] * degree_nodewise[col[i]])
        ) - np.log(k)
        score = 0 if score < 0 else score
        pmi_matrix[row[i], col[i]] = score

    return pmi_matrix


def get_dynamics(pmi_matrix: np.array, b_history: np.array, f_history: np.array, top_u: float) -> np.array:
    """provide decision on whether a node is dynamic or status by
    evaluating how well the history feature fits current topology
    Args:
        top_u:
        f_history:
        b_history:
        pmi_matrix (np.array): pmi matrix

    Returns:
        np.array: p_s and pd provide description the dynamics of current graph compared to the history.
    """
    node_num = pmi_matrix.shape[0]
    history_approximate = b_history @ f_history
    error = pmi_matrix - history_approximate
    delta_nodewise = np.sum(error * error, axis=0)
    # ascending  sort 
    sort_dynamics = np.argsort(delta_nodewise)
    statics = sort_dynamics[0:int(np.ceil(node_num * top_u))]

    p_static = np.zeros((node_num, node_num),dtype=float)
    for i in range(len(statics)):
        p_static[sort_dynamics[i], sort_dynamics[i]] = 1
    p_dynamic = np.eye(node_num,dtype=float) - p_static
    return p_static, p_dynamic


def spectral_clustering(x: np.array, n_cluster: int) -> np.array:
    """
    
    Args:
        x (np.array): feature matrix $x /in R^{N times D}$
        n_cluster (int): cluster number

    Returns:
        np.array: clustering labels
    """
    model = SpectralClustering(n_clusters=n_cluster,
                               assign_labels='discretize',
                               random_state=0).fit(x)
    labels = model.labels_
    partition = [[] for i in range(n_cluster)]
    for i in range(x.shape[0]):
        partition[labels[i]].append(i + 1)

    """grids = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if model.labels_[i] == model.labels_[j]:
                grids[i,j] = 1"""
    return partition


def comput_confusionmatrix(c1: np.array, c2: np.array) -> np.array:
    """compute confusion matrix for two clustering results.

    Args:
        c1 (np.array): clustering result 1
        c2 (np.array): clustering result 2
    Returns:
        np.array: confusion matrix
    """
    len_c1, len_c2 = len(c1), len(c2)
    confusion_matrix = np.zeros((len_c1, len_c2))
    for i in range(len_c1):
        for j in range(len_c2):
            confusion_matrix[i, j] = len(set(c1[i]).intersection(set(c2[j])))
    return confusion_matrix


def compute_nmi(confusion_matrix: np.array) -> float:
    # TODO add math description in comments
    """compute normalized mutual information for input pair of confusion matrix.

    Args:
        confusion_matrix (np.array): confusion_matrix

    Returns:
        float: nmi score 
    """
    smallest = np.finfo(np.float64).tiny
    n_samples = np.sum(confusion_matrix) + 1
    mutual_information = 0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            mutual_information += confusion_matrix[i, j] * np.log(
                confusion_matrix[i, j] * n_samples /
                (np.sum(confusion_matrix[i, :]) * np.sum(confusion_matrix[:, j]) + smallest)
                + smallest)
    h1 = 0
    h2 = 0
    for i in range(confusion_matrix.shape[0]):
        n_i = np.sum(confusion_matrix[i, :])
        h1 += n_i * np.log(n_i / n_samples + smallest)
    for j in range(confusion_matrix.shape[1]):
        n_j = np.sum(confusion_matrix[:, j])
        h2 += n_j * np.log(n_j / n_samples + smallest)
    nmi_score = -2 * mutual_information / (h1 + h2)
    return nmi_score

def evaluate_clustering(features,partition_groundtruth):
    n_clusters = len(partition_groundtruth)
    print("n_clusters: ", n_clusters)
    partition = spectral_clustering(x=features,
                                    n_cluster=n_clusters)
    confusion_matrix = comput_confusionmatrix(partition_groundtruth, partition)
    nmi_score = compute_nmi(confusion_matrix)
    return partition,nmi_score
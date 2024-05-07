"""
Author: Ma Yaxiong
E-mail: mayaxiong@xidian.edu.cn
Description:  to be added
Time: 2024/1/21
Filename: dataloader.py
"""
import os
import numpy as np
import scipy.io as scio
import h5py
import time
from collections import Counter

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_gn_graph(prefix='synfix_3', num_snapshots=10):
    data_dir = os.path.join('dataset', 'GN')
    pre = prefix
    max_node_num = 0
    partition = []
    # load adjacency_matrices and ground truth
    for i in range(num_snapshots):
        partition_ = []
        step = str(i + 1).zfill(2) if i + 1 < 10 else str(i + 1).zfill(3)
        with open(os.path.join(data_dir, pre + '.' + 't' + step + '.comm')) as f:
            for line in f.readlines():
                partition_.append([int(s) for s in line.strip().split()])
        node_num = max([max(seq) for seq in partition_])
        max_node_num = max(max_node_num, node_num)
        partition.append(partition_)
    print(max_node_num)
    adjacency_matrices = np.zeros((num_snapshots, max_node_num, max_node_num))
    for i in range(num_snapshots):
        step = str(i + 1).zfill(2) if i + 1 < 10 else str(i + 1).zfill(3)
        with open(os.path.join(data_dir, pre + '.' + 't' + step + '.edges')) as f:
            for line in f.readlines():
                src, dst = map(int, line.strip().split())
                adjacency_matrices[i, src - 1, dst - 1] = 1

    # load ground truth

    return adjacency_matrices, partition


def load_data_from_mat(path_data='path_data'):
    """

    Args:
        prefix:

    Returns:

    """
    res = {}

    print(f"IS PATH_DATA EXISTED:{os.path.exists(path_data)}")
    try:
        data = scio.loadmat(path_data)["data"]
        data = np.array(data).transpose(2,0,1)
    except NotImplementedError:
        print("h5py used in data loading")


        data =  load_from_mat_(path_data)
        # no need to transpose cause h5py do it atumatically
        data = np.array(data)
    return data

def load_label_from_mat(path_label = "path_label"):
    """

    Args:
        prefix:

    Returns:

    """
    print(f"IS PATH_LABEL EXISTED:{os.path.exists(path_label)}")
    try:

        label = np.array(scio.loadmat(path_label)["label"])

    except NotImplementedError:
        print("h5py used in label loading")
        label  = load_from_mat_(path=path_label)
        label = label.T
    return label



def load_from_mat_(path="path"):
    """load temporary graph from .mat data
    Args:
        prefix:
        is_real:
    Returns:
    """

    # path_data = os.path.join('dataset', f'{prefix}.mat')
    f = h5py.File(path, 'r')

    data = f.get(list(f.keys())[0])
    data = np.array(data)
    f.close()
    return data


def get_partition_groundtruth_from_label(labels: np.array) -> np.array:
    """

    Args:
        labels:

    Returns:

    """
    partition = []
    max_label = int(np.max(labels))
    # snapshot wise
    for i in range(labels.shape[0]):
        label_set = list(set(labels[i, :]))
        n_cluster = len(label_set)
        partition_ = [[] for s in range(n_cluster)]
        for k in range(labels.shape[1]):
            partition_[label_set.index(labels[i, k])].append(k+1)
        partition.append(partition_)
    return partition

def process_labels(labels):
    # labels  1XN
    threshold = int(labels.shape[0]*0.8)
    sorted_labels = sorted(Counter(labels).items(), key=lambda x:x[1],reverse=True)
    adopt_label = []
    current_sum = 0
    for i in range(len(sorted_labels)):
        current_sum = current_sum + sorted_labels[i][1]
        if current_sum >= threshold:
            break
        adopt_label.append(sorted_labels[i][0])
    for k in range(labels.shape[0]):
        if labels[k] not in adopt_label:
            # 0 for scare cluster
            labels[k] = 0
    return labels

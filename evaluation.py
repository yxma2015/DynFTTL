# -*-coding:utf-8 -*-

"""
# File       : evaluation.py
# Author     ：Yaxiong Ma
# E-Mail    : mayaxiong@xidian.edu.cn
# Create Time: 2024/3/22 14:14
# Description：Evaluation clustering proformance based on learned features and "the ground truth" labels"
"""
import os
import numpy as np
from utils.utils import spectral_clustering, comput_confusionmatrix, compute_nmi,compute_pmi_matrix,get_laplace_matrix
from utils.dataloader import  load_label_from_mat,get_partition_groundtruth_from_label,process_labels

def evaluate_clustering(features,partition_groundtruth):
    n_clusters = len(partition_groundtruth)
    #n_clusters = 8
    partition = spectral_clustering(x=features,
                                    n_cluster=n_clusters)
    confusion_matrix = comput_confusionmatrix(partition_groundtruth, partition)
    nmi_score = compute_nmi(confusion_matrix)
    return partition,nmi_score
#  for test

root_path_feature = "root_path_feature"

root_path_label = "root_path_label"

dataset_list = ["synfix_3","synvar_3","hide","expand","mergesplit","birthdeath","email","cellphone","wikipedia","dublin"]


all_nmi_scores = {}
all_partitions = {}

path_nmi_scores = "path_nmi_scores"
path_partitions = "path_partitions"
if os.path.exists(path_nmi_scores):
    all_nmi_scores = np.load(path_nmi_scores,allow_pickle=True).all()
    print(all_nmi_scores)
if os.path.exists(path_partitions):
    all_partitions = np.load(path_partitions,allow_pickle=True).all()

for data in all_nmi_scores.keys():
     dataset_list.remove(data)
print(dataset_list)
for data in dataset_list:
    print(f"data:{data}")
    labels = load_label_from_mat(path_label=
                                 os.path.join(
                                     root_path_label,
                                     f"{data}_label.mat"))
    if data in ["dublin","wikipedia"]:
        for k in range(labels.shape[0]):
            labels[k] = process_labels(labels[k])
    partition_groundtruth = get_partition_groundtruth_from_label(labels)
    print(f"partition_groundth of data:{data} get!")


    nmi_scores = []
    partitions = []

    for step in range(1,labels.shape[0]+1):
        print(f"{data}:step{step}")
        features = np.load(os.path.join(root_path_feature, f"{data}_step{step}_feature.npy")).T
        partition,nmi_score = evaluate_clustering(features,partition_groundtruth[step-1])
        nmi_scores.append(nmi_score)
        partitions.append(partition)
    mean_nmi_score = np.mean(nmi_scores)
    print(f"nmi_scpres_{data}: {nmi_scores}\nmean_nmi_scpres: {mean_nmi_score}")
    all_nmi_scores[data]=nmi_scores
    all_partitions[data]=partitions
    np.save(path_nmi_scores, all_nmi_scores)
    np.save(path_partitions, all_partitions)
print(all_nmi_scores)


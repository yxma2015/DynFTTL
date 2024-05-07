
# -*-coding:utf-8 -*-

"""
# File       : test_torch_stepwise_0325_modified.py
# Author     ：Yaxiong Ma
# E-Mail    : mayaxiong@xidian.edu.cn
# Create Time: 2024/3/25 10:04
# Description："update dynamics evaluating"
"""
import os

import gc
import pickle
import h5py
import torch
import torch.optim as optim
import numpy as np
from utils.dataloader import load_gn_graph, load_data_from_mat,load_label_from_mat,get_partition_groundtruth_from_label
from utils.utils import spectral_clustering, comput_confusionmatrix, compute_nmi,compute_pmi_matrix,get_laplace_matrix
# from model import DynamicGraphAnalysis
from model_withtorch_stepwise_auto_modified_0325 import ModelStepwise
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler

# n cluster 4 4 50 50 50 50
data_set_root_path = 'dataset'

dataset_list = ["synfix_3","synvar_3","hide","expand","mergesplit","birthdeath","email","cellphone","wikipedia","dublin"]
# recomended
dimension_list = [64,64,256,256,256,256,256,256,256,256]
dataset_list = ["cellphone"]
dimension_list = [128]
# recomended
# dimension_list = [64,64,256,256,256,256,256,256,64]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scores_0323 =  {}
def test(data,dim):
    print(f"{data}")
    dataset = data
    dimension = dim

    epochs = 200
    print("data preparation")
    path_data = os.path.join(data_set_root_path,data+".mat")
    path_label = os.path.join(data_set_root_path,data+"_label.mat")
    adjacency_matrices = load_data_from_mat(path_data=path_data)
    adjacency_matrices = adjacency_matrices
    labels = load_label_from_mat(path_label=path_label)
    print("pls check:",adjacency_matrices.shape,labels.shape)
    #adjacency_matrices, labels = np.array(data).astype(np.float32), np.array(label)
    #del data,label
    #gc.collect()

    timespan = adjacency_matrices.shape[0]
    node_num = adjacency_matrices.shape[1]
    pmi_matrices = np.empty((timespan,
                             node_num,
                            node_num),dtype=np.float32)
    lap = np.empty((timespan,
                    node_num,
                    node_num),dtype=np.float32)

    for i in range(timespan):
        pmi_matrices[i,:,:] = compute_pmi_matrix(adjacency_matrices[i,:,:],k=2)
        lap[i,:,:] = get_laplace_matrix(pmi_matrices[i,:,:])
    partition_groundtruth = get_partition_groundtruth_from_label(labels)

    loss_history = np.zeros((epochs, timespan, 5))
    history_feature = torch.zeros((dimension, node_num),dtype=torch.float32)
    history_base = torch.zeros((node_num, dimension), dtype=torch.float32)
    history_pmi_matrix = torch.zeros((node_num, node_num),dtype=torch.float32)
    nmi_scores = np.zeros(timespan)
    for step in range(timespan):
        is_initialstep = True if step == 0 else False
        analyzer = ModelStepwise(parameters={
            "is_initial_step": is_initialstep,
            "device":device,
            "dimension":dimension,
            "node_number":node_num,
            "alpha":0.1,
            "beta":0.1,
            "epochs":epochs,
            "top_u":0.1,
            "history_feature":history_feature,
            "history_base":history_base}).to(device)
        #  keep memory of feature and base matrix when learning
        lr=0.1
        # optimizer = optim.SGD(analyzer.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(analyzer.parameters(), lr=0.1,eps=1e-4)
        #scheduler = lr_scheduler.LinearLR(optimizer, start_factor=10, end_factor=1e-3, total_iters=epochs)

        loss_old = 0
        s = 0
        for k in range(analyzer.epochs):
            if not analyzer.is_initial_step:
                history_pmi_matrix = torch.from_numpy(pmi_matrices[step-1,:,:])


            optimizer.zero_grad()
            loss = analyzer.forward(pmi_matrices=torch.from_numpy(pmi_matrices[step,:,:]).to(device),
                                    lap_matrices=torch.from_numpy(lap[step,:,:]).to(device),
                                    history_feature=history_feature.to(device),
                                    history_pmi_matrix=history_pmi_matrix.to(device))
            loss_history[k,step,:] = loss.detach().to("cpu")
            # print("loss:",loss)
            #for t in range(analyzer.timespan):
                #analyzer.base_list[t] = torch.nn.functional.relu(analyzer.base_list[t])
                #analyzer.feature_list[t] = torch.nn.functional.relu(analyzer.feature_list[t])
            loss = torch.sum(loss)
            delta_loss = loss_old - loss.detach().to("cpu")
            print(f"\rEpoch {k + 1} for step {step +1}'s loss is:{loss}",end=' ')

            if 0 < delta_loss <5e-4:
                s += 1
                if s == 6:
                    print(f"early stopping at epoch {k}")
                    break
            else:
                s = 0
            loss_old = loss.detach().to("cpu")
            loss.backward()
            optimizer.step()

        history_feature = analyzer.feature.detach()
        history_base = analyzer.base.detach()

        np.save(f"./EXPERIMENTS/{dataset}_step{step+1}_feature.npy",
                analyzer.feature.detach().cpu().numpy())
        np.save(f"./EXPERIMENTS/{dataset}_step{step + 1}_base.npy",
                analyzer.base.detach().cpu().numpy())
        if not analyzer.is_initial_step:
            np.save(f"./EXPERIMENTS/{dataset}_step{step+1}_p_d.npy",
                    analyzer.p_d.detach().cpu().numpy())
            np.save(f"./EXPERIMENTS/{dataset}_step{step+1}_transition.npy",
                    analyzer.transition.detach().cpu().numpy())
        # print(analyzer.memory["Q_D"][i, :, :])
    np.save(os.path.join("EXPERIMENTS",f"{dataset}_loss_history.npy"), loss_history)
for d in range(len(dataset_list)):
    test(data=dataset_list[d],dim=dimension_list[d])
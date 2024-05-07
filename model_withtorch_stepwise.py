# -*-coding:utf-8 -*-

"""
# File       : model_withtorch_stepwise_auto_modified_0325.py
# Author     ：Yaxiong Ma
# E-Mail    : mayaxiong@xidian.edu.cn
# Create Time: 2024/3/25 10:04
# Description：model building
        230325:update dynamics evaluating rules to |m_{[t-1]} - B^{[t]}F^{[t]}|
"""
import numpy as np
import tqdm
from utils.utils import relu, get_laplace_matrix, nmf, compute_pmi_matrix
import torch
import torch.nn as nn

class ModelStepwise(nn.Module):
    def __init__(self, parameters):
        super(ModelStepwise, self).__init__()
        self.device = parameters['device']
        self.is_initial_step = parameters['is_initial_step']
        self.dimension = nn.Parameter(
            torch.tensor(parameters['dimension']), requires_grad=False)
        self.node_number = nn.Parameter(
            torch.tensor(parameters['node_number']), requires_grad=False)
        self.alpha = nn.Parameter(
            torch.tensor(parameters['alpha']), requires_grad=False)
        self.beta = nn.Parameter(
            torch.tensor(parameters['beta']), requires_grad=False)
        self.epochs = nn.Parameter(
            torch.tensor(parameters['epochs']), requires_grad=False)
        self.top_u = nn.Parameter(
            torch.tensor(parameters['top_u']), requires_grad=False)
        self.base = nn.Parameter(torch.rand((self.node_number, self.dimension),
                                            dtype=torch.float32)
                                 )
        self.feature = nn.Parameter(torch.rand((self.dimension,
                                                self.node_number)
                                               )
                                     )
        if not self.is_initial_step:
            self.transition = nn.Parameter(torch.rand((self.node_number, self.node_number),
                                                       dtype=torch.float32))
            self.q_s = torch.eye(self.node_number, dtype=torch.float32).to(self.device)
            self.p_s = torch.zeros((self.node_number, self.node_number), dtype=torch.float32).to(self.device)
            self.p_d = torch.zeros((self.node_number, self.node_number), dtype=torch.float32).to(self.device)
            self.base = nn.Parameter(parameters["history_base"])
            self.feature = nn.Parameter(parameters["history_feature"])

    def objective_function(self,
                           input_pmimatrices,
                           input_lap,
                           history_feature,
                           history_pmi_matrix):
        """

        Args:
            input:

        Returns:

        """
        loss_component = torch.zeros(5,dtype=torch.float32)


        loss_component = self.compute_loss(
            pmi_matrix = input_pmimatrices,
            lap_matrix = input_lap,
            history_feature = history_feature,
            history_pmi_matrix=history_pmi_matrix)
        return loss_component

    def compute_loss(self,
                     pmi_matrix,
                     lap_matrix,
                     history_feature,
                     history_pmi_matrix):
        # TODO 完善损失函数
        loss = torch.zeros(5,dtype=torch.float32)
        loss[0] = self.loss_0(m=pmi_matrix,
                              b=self.base,
                              f=self.feature
                              )
        loss[4] = self.loss_4(f=self.feature,
                              lap=lap_matrix)
        if not self.is_initial_step:
            self.p_s,self.p_d = self.get_dynamics(history_pmi_matrix=history_pmi_matrix,
                                   feature=self.feature,
                                   base=self.base,
                                   top_u=self.top_u)
            loss[1] = self.loss_1(f_history=history_feature,
                                  q_s=self.q_s,
                                  q_d=self.transition,
                                  f=self.feature)
            loss[2] = self.loss_2(f_history=history_feature,
                                  p_s=self.p_s,
                                  q_s=self.q_s,
                                  f=self.feature)
            loss[3] = self.loss_3(f_history=history_feature,
                                  p_d=self.p_d,
                                  q_d=self.transition,
                                  f=self.feature)
        return loss
    def loss_0(self,m,b,f):
        return torch.norm(m - b @ f) / self.node_number
    def loss_1(self,f_history,q_s,q_d,f):
        return self.alpha * torch.norm(f_history @ (q_s + q_d) - f) /self.node_number
    def loss_2(self,f_history,p_s,q_s,f):
        return self.alpha * torch.norm(f_history @ p_s @ q_s - f @ p_s) /self.node_number
    def loss_3(self,f_history,p_d,q_d,f):
        return self.alpha * torch.norm(f_history @ p_d @ q_d - f @ p_d) /self.node_number
    def loss_4(self,f,lap):
        return torch.trace(f @ lap @ f.T) /self.node_number

    def forward(self,pmi_matrices,lap_matrices,history_feature,history_pmi_matrix):
        return self.objective_function(pmi_matrices,lap_matrices,history_feature,history_pmi_matrix)


    def compute_pmi_matrix(self,adjacency_matrix: torch.tensor, k: torch.int) -> torch.tensor:
        """

        Args:
            adjacency_matrix: adjacency matrix of a snapshot graph
            k: constant

        Returns:

        """

        degree_nodewise = torch.sum(adjacency_matrix, axis=0)
        num_edge = torch.sum(degree_nodewise)
        row, col = torch.nonzero(adjacency_matrix)
        pmi_matrix = adjacency_matrix
        for i in range(len(row)):
            score = torch.log(
                adjacency_matrix[row[i], col[i]] *
                num_edge / (degree_nodewise[row[i]] * degree_nodewise[col[i]])
            ) - torch.log(k)
            score = 0 if score < 0 else score
            pmi_matrix[row[i], col[i]] = score

        return pmi_matrix


    def get_dynamics(self,history_pmi_matrix: torch.tensor, base: torch.tensor, feature: torch.tensor, top_u: torch.float) -> torch.tensor:
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
        # history_approximate = b_history @ f_history
        current_approximate = base @ feature
        error = history_pmi_matrix - current_approximate
        delta_nodewise = torch.sum(error * error, axis=0)
        # ascending  sort
        sort_dynamics = torch.argsort(delta_nodewise)
        statics = sort_dynamics[0:int(torch.ceil(self.node_number * (1-top_u)))]

        p_static = torch.zeros((self.node_number, self.node_number),dtype=torch.float32).to(self.device)
        p_dynamic = torch.zeros((self.node_number, self.node_number),dtype=torch.float32).to(self.device)
        for i in range(len(statics)):
            p_static[sort_dynamics[i], sort_dynamics[i]] = 1
        p_dynamic = (torch.eye(self.node_number,dtype=torch.float32).to(self.device) - p_static)
        return p_static, p_dynamic





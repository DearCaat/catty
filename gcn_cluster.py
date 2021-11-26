###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import networkx as nx

def connected_component(edges,scores,thr=0.75):
    score_dict = {} # score lookup table
    new_graph=list()
    for i,e in enumerate(edges):
        if scores[i]>=thr:
            score_dict[e[0], e[1]] = scores[i]
    G = nx.Graph()
    nodes=set(edges[:,0])
    G.add_nodes_from(nodes)
    G.add_edges_from(score_dict.keys())
    for c in nx.connected_components(G):
        new_graph.append(c)
    return new_graph

def cluster_feat(vectors,edges,scores):
    new_feat=[]
    new_graph=connected_component(edges,scores)
    for i in new_graph:
        ips_graph=list()
        for j in i:
            ips_graph.append(vectors[j])
        ips_graph = torch.stack(ips_graph,0)
        ips_graph=torch.mean(ips_graph,dim=0)
        new_feat.append(ips_graph)
    new_feature = torch.stack(new_feat,0)
    return new_feature

def ConsineDistance(vectors):
    """
    Implementation of adjacency matrix
    """
    B,N,D = vectors.shape
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    vecs_i = vectors.unsqueeze(1).repeat(1, N, 1, 1)                # (B, N , N, feature_dim)
    vecs_j = vectors.unsqueeze(2).repeat(1, 1, N, 1)                # (B, N , N, feature_dim)
    similarity = cosine_similarity(vecs_i, vecs_j)
    # sort_index = torch.argsort(similarity, dim=-1, descending=True)[:, 1:k+1]
    # A = torch.zeros(N,N)
    # A[np.repeat(np.arange(N), k), sort_index.flatten()] = 1
    return similarity   # (B, N,N)

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

def normalize_adj(A, type="AD"):
    if type == "DAD":
        # d is  Degree of nodes A=A+I
        # L = D^-1/2 A D^-1/2
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = D - A
    return G

def reshape_A(edges,scores,N):
    score_dict = torch.ones(N,N)
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = scores[i]
    # a = torch.from_numpy(score_dict)
    # a = a.reshape(N,-1)
    # b = torch.ones(N, N)
    # b[:-1,1:] += torch.triu(a[:-1])
    # b[1:,:-1] += torch.tril(a[1:])
    return score_dict

def part_feat(vectors,edges,scores):
    new_feat=[]
    new_graph=connected_component(edges,scores)
    for i in new_graph:
        ips_graph=list()
        for j in i:
            ips_graph.append(vectors[j])
        ips_graph = torch.stack(ips_graph,0)
        # ips_graph=torch.mean(ips_graph,dim=0)
        new_feat.append(ips_graph)
    # new_feature = torch.stack(new_feat,0)
    return new_feat

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, features, A ):
        x = torch.bmm(A, features)
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(features,A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out 
        

class GCN(nn.Module):
    def __init__(self,in_dim=512,out_dim=256,k1=20):
        super(GCN, self).__init__()
        self.k1 = k1
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = GraphConv(in_dim, in_dim, MeanAggregator)
        self.conv2 = GraphConv(in_dim, in_dim, MeanAggregator)
        self.conv3 = GraphConv(in_dim, out_dim, MeanAggregator)
        self.conv4 = GraphConv(out_dim, out_dim,MeanAggregator)
        
        self.classifier = nn.Sequential(
                            nn.Linear(out_dim, out_dim),
                            nn.PReLU(out_dim),
                            nn.Linear(out_dim, 2))
    
    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B,N,I,D = x.shape    # batch_size instance_size nodes_IPS dim
        #xnorm = x.norm(2,2,keepdim=True) + 1e-8
        #xnorm = xnorm.expand_as(x)
        #x = x.div(xnorm)
        
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B,N,I,D)

        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A)

        edge_feat = x[one_hop_idcs].view(B,N,self.k1,D)
        edge_feat = edge_feat.view(B,-1,D)
        pred = self.classifier(edge_feat)
            
        # shape: B (N*k1) 2
        return pred

class KnnGraph(object):
    def __init__(self,active_connection=4,k_at_hop=[20,5],distance='cosine'):
        self.active_connection = active_connection
        self.k_at_hop = k_at_hop
        self.distance = distance
        
    def get_KNN(self,feats,distance='cosine'):
        if distance == 'cosine':
            similarity_matrix=ConsineDistance(feats)  # B*N*N
        elif distance == 'Euclidean':
            similarity_matrix=EuclideanDistances(feats,feats)
        knn_graph = torch.argsort(similarity_matrix, axis=2,descending=True)  # N*N
        return knn_graph

    def __call__(self, feats, gt_data=None):
        B,N,D=feats.shape

        # ## 1. get the KNN graph
        knn_graph = self.get_KNN(feats,self.distance)
        knn_graph = knn_graph[:,:, :self.k_at_hop[0] + 1]

        # 添加第一跳
        hops = knn_graph[:,:,:]      # B N [1 K1] 这里的1是center point 的索引
        hops_1 = hops.clone()
        hops_2 = hops.unsqueeze(-1).repeat(1,1,1,self.k_at_hop[1]+1).clone()  #B N [1 K1] [1 K2]
        del hops
        # 添加第二跳
        for i in range(B):
            for m in range(N):
                hops_2[i,m,:,:] = hops_1[i,hops_1[i,m,:],:self.k_at_hop[1]+1]   
        del hops_1
        # hops_2矩阵中，dim -1 第一个元素存储的是K1跳的顶点，后几个元素为K2跳个顶点，dim -2 的第一个元素是中心点的索引
        # 展平hops矩阵后两维，这里就不考虑自身的那一维
        uni = hops_2[:,:,1:,:].flatten(start_dim=-2, end_dim=-1)

        # 构建唯一顶点矩阵 构建邻接矩阵 因为每一个lps的顶点数目都不相同，因此需要使用for 循环
        uni_array = np.empty((B,N),dtype=object)
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        A_ = torch.zeros(B,N,max_num_nodes,max_num_nodes)
        feat = torch.zeros(B,N,max_num_nodes,D)
        mask_one_hop_idcs = torch.zeros(32,49,max_num_nodes) == 1
        for i in range(B):
            for m in range(N):
                uni_tmp = torch.unique(uni[i,m])
                if m not in uni_tmp:
                    uni_tmp = torch.cat((torch.tensor([m]),uni_tmp))
                uni_array[i,m] = uni_tmp
                num_nodes = len(uni_tmp)
                a_tmp = torch.zeros(size=(num_nodes,num_nodes))
                neighbors = knn_graph[i,uni_array[i,m],1:self.active_connection+1]
                # 每个结点能够连接的顶点数不同，需要使用for循环
                for node in range(num_nodes):
                    nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
                    a_tmp[node,nei_index] = 1
                    a_tmp[nei_index,node] = 1
                # one-hop indices
                mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops[i,m,1:,0])
                A_[i,m,:num_nodes,:num_nodes] = a_tmp      
                feat[i,m,:num_nodes] = feats[i,uni_tmp]

        # 正则化邻接矩阵？
        D = A_.sum(-1,keepdim=True)
        A_ = A_.div(D)
        A_[torch.isnan(A_)] = 0

        # 正则化特征，IPS特征减去中心点特征
        feat = feat - feats.unsqueeze(-2)

        return feat,A_,mask_one_hop_idcs
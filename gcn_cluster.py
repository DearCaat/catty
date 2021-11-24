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

def connected_component(edges,scores):
    score_dict = {} # score lookup table
    new_graph=list()
    for i,e in enumerate(edges):
        if scores[i]>=0.75:
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
    vecs_i = vectors.unsqueeze(0).repeat(1, N, 1, 1)                # (B, N , N, feature_dim)
    vecs_j = vectors.unsqueeze(1).repeat(1, 1, N, 1)                # (B, N , N, feature_dim)
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
    def __init__(self):
        super(GCN, self).__init__()
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        self.conv1 = GraphConv(512, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256,MeanAggregator)
        
        self.classifier = nn.Sequential(
                            nn.Linear(256, 256),
                            nn.PReLU(256),
                            nn.Linear(256, 2))
    
    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B,N,D = x.shape
        #xnorm = x.norm(2,2,keepdim=True) + 1e-8
        #xnorm = xnorm.expand_as(x)
        #x = x.div(xnorm)
        
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B,N,D)

        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B,k1,dout).cuda()
        for b in range(B):
            edge_feat[b,:,:] = x[b, one_hop_idcs[b]]  
        edge_feat = edge_feat.view(-1,dout)
        pred = self.classifier(edge_feat)
            
        # shape: (B*k1)x2
        return pred

class KnnGraph(object):
    def __init__(self,active_connection):
        # self.k_at_hop = k_at_hop
        # self.depth = len(self.k_at_hop)
        self.active_connection = active_connection
        # self.cluster_threshold = 0.75

    def localIPS(self, k_at_hop, knn_graph):
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        knn_graph = knn_graph[:, :, :k_at_hop[0] + 1]
        depth = len(k_at_hop)
        hops_list = list()
        one_hops_list = list()
        for index, cluster in enumerate(knn_graph):
            hops = list()
            center_idx = index

            h0 = set(knn_graph[center_idx][1:])
            hops.append(h0)
            # Actually we dont need the loop since the depth is fixed here,
            # But we still remain the code for further revision
            for d in range(1, depth):
                hops.append(set())
                for h in hops[-2]:
                    hops[-1].update(set(knn_graph[h][1:k_at_hop[d] + 1]))

            hops_set = set([h for hop in hops for h in hop])
            nodes_list = list(hops_set)
            nodes_list.insert(0, center_idx)
            hops_list.append(nodes_list)
            one_hops_list.append(h0)
        # shape B*N
        return hops_list,  one_hops_list

    def graph_IPS(self,k_at_hop,feature_bin,knn_graph_bin,hops_bin, one_hops_bin):
        '''
        输入：fature,knn_graph
        输出：每个图的特征、邻接矩阵、一跳节点
        '''
    
        feat_batch = list()
        adj_batch = list()
        cid_batch=list()
        h1id_batch = list()
        unique_ips_batch=list()

        for index, cluster in enumerate(hops_bin):
            # feat_map=feature_bin[index]
            num_nodes=int(len(cluster))  # 邻居节点数量n，每个图的n是不一定大小的
            center_idx=cluster[0]
            one_hops_list=one_hops_bin[index]  # 每个节点的一阶节点

            # IPS内部排序
            unique_nodes_map = {j: i for i, j in enumerate(cluster)}
            center_node = torch.tensor([unique_nodes_map[center_idx], ]).type(torch.long)
            one_hop_idcs = torch.tensor([unique_nodes_map[i] for i in one_hops_list],dtype=torch.long)

            center_feat = feature_bin[torch.tensor(center_idx, dtype=torch.long)]
            feat = feature_bin[torch.tensor(cluster, dtype=torch.long)]  # n*D
            feat = feat - center_feat  # 节点特征归一化n*D

            # max_num_nodes = max([len(ips) for hops in hops_bin for ips in hops])
            max_num_nodes = k_at_hop[0] * (k_at_hop[1] + 1) + 1  # 每个图最多有max_num_nodes个节点
            A = np.zeros((num_nodes, num_nodes))  #n*n
            feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, feat.shape[1]).cuda()], dim=0)  # max_num_nodex*D
            # feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, feat.shape[1])], dim=0)  # max_num_nodex*D

            for node in cluster:
                neighbors = knn_graph_bin[node, 1:self.active_connection + 1]
                for n in neighbors:
                    if n in cluster:
                        A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                        A[unique_nodes_map[n], unique_nodes_map[node]] = 1  #A-->n*n

            A = normalize_adj(A, type="DAD")
            # D = A.sum(1)
            # A = A.div(D)
            A_ = torch.zeros(max_num_nodes, max_num_nodes)
            A_[:num_nodes, :num_nodes] = A
            # Testing
            unique_ips = torch.tensor(cluster)
            unique_ips = torch.cat([unique_ips, torch.zeros(max_num_nodes - num_nodes, dtype=torch.long)], dim=0)

            feat_batch.append(feat) # B*m*d-->m是固定的
            adj_batch.append(A_)  # B*m*m
            cid_batch.append(center_node)
            h1id_batch.append(one_hop_idcs)  # B*n1
            unique_ips_batch.append(unique_ips)
        
        feat_bth = torch.stack(feat_batch,0).cuda()
        adj_bth = torch.stack(adj_batch,0).cuda()
        cid_bth = torch.stack(cid_batch, 0).cuda()
        h1id_bth = torch.stack(h1id_batch,0).cuda()
        unique_node_bth = torch.stack(unique_ips_batch, 0).cuda()
        return feat_bth, adj_bth, cid_bth, h1id_bth,unique_node_bth

    def __call__(self, feats, gt_data=None):
        B,N,D=feats.shape
        if N<=20:
            k_at_hop=[N,5]
        else:
            k_at_hop = [20, 5]
        # ## 1. computing cosine similarity of Node feature
        similarity_matrix=ConsineDistance(feats)  # B*N*N
        # similarity_matrix=EuclideanDistances(feats,feats)
        # knn_graph = np.argsort(-similarity_matrix, axis=1)[:, :]
        # ## 2. compute the knn graph
        # knn_graph = torch.argsort(similarity_matrix, axis=1,descending=False)[:, :]
        knn_graph = torch.argsort(similarity_matrix, axis=2,descending=True)  # B*N*N
        knn_graph=knn_graph.cpu().numpy()
        # knn_graph=knn_graph.cuda()
        hops, one_hops = self.localIPS(k_at_hop, knn_graph)  # B*邻居节点数
        # ## 3.
        feat_bth, adj_bth, cid_bth,h1id_bth,unique_node_bth = self.graph_IPS(k_at_hop,feats,knn_graph,hops, one_hops)

        return feat_bth, adj_bth, cid_bth,h1id_bth,unique_node_bth
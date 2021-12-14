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
import torch.multiprocessing as mp

def gcn_cluster(edge_col_indices,scores,feat,thr=0.75):
    B,N,K1 = edge_col_indices.shape
    #if scores
    crow_indices = torch.tensor([K1*i for i in range(N+1)]).contiguous().cuda()
    cluster_feat = [[] for i in range (B)]
    cluster_idcs = [[] for i in range (B)]
    for b in range(B):
        csr = torch.sparse_csr_tensor(crow_indices,edge_col_indices[b,:,:].flatten().contiguous(),scores[b,:,:,0].flatten().contiguous(),size=(N,N))
        A = csr.to_dense()
        A[A>thr] = 1
        A_nx = nx.from_numpy_matrix(A.cpu().numpy())
        for c in nx.connected_components(A_nx):
            c = list(c)
            cluster_feat[b].append(feat[b][c])
            cluster_idcs[b].append(c)
    del crow_indices
    return cluster_feat,cluster_idcs

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

def EuclideanDistances(vectors):
    B,N,D = vectors.shape
    pdist = torch.nn.PairwiseDistance()
    a = vectors.unsqueeze(1).repeat(1, N, 1, 1)                # (B, N , N, feature_dim)
    b = vectors.unsqueeze(2).repeat(1, 1, N, 1)                # (B, N , N, feature_dim)
    similarity = pdist(a, b)
    return similarity

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

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, features, A ):
        #x = torch.bmm(A, features)
        x = torch.einsum('bnii,bnid->bnid',(A, features))
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
        b, n, i, d = features.shape
        assert(d==self.in_dim)
        out = self.agg(features,A)
        out = torch.cat([features, out], dim=3)
        out = torch.einsum('bnid,df->bnif', (out, self.weight))
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

        dim_out = x.size(-1)
        edge_feat = x[one_hop_idcs].view(B,N,self.k1,dim_out)
        edge_feat = edge_feat.view(-1,dim_out)
        pred = self.classifier(edge_feat).view(B,N,self.k1,2)
            
        # shape: B N k1 2
        return pred

class KnnGraph(object):
    def __init__(self,active_connection=4,k_at_hop=[20,5],distance='cosine'):
        self.active_connection = active_connection
        self.k_at_hop = k_at_hop
        self.distance = distance
    def get_KNN(self,feats,distance='cosine'):
        if distance == 'cosine':
            similarity_matrix=ConsineDistance(feats)  # B*N*N
            knn_graph = torch.argsort(similarity_matrix, axis=2,descending=True)  # B*N*N
        elif distance == 'euclidean':
            similarity_matrix=EuclideanDistances(feats)
            knn_graph = torch.argsort(similarity_matrix, axis=2,descending=False)  # B*N*N
        return knn_graph
    
    #for multi-process
    def change_hops(self,B,N,hops_1,hops_2,knn_graph,mask_one_hop_idcs,A_,feat,feats):
        for i in B:
            for m in range(N):
                hops_2[i,m,:,:] = hops_1[i,hops_1[i,m,:],:self.k_at_hop[1]+1]
                #展平hops矩阵后两维，这里就不考虑自身的那一维
                uni_tmp = hops_2[i,m,1:,:].flatten(start_dim=-2, end_dim=-1)
                uni_tmp = torch.unique(uni_tmp)
                if m not in uni_tmp:
                    uni_tmp = torch.cat((torch.tensor([m]).cuda(non_blocking=True),uni_tmp))
                num_nodes = len(uni_tmp)
                a_tmp = torch.zeros(num_nodes,num_nodes) # 小维度的矩阵cpu上进行索引和改值更快一点
                neighbors = knn_graph[i,uni_tmp,1:self.active_connection+1]
                # 每个结点能够连接的顶点数不同，需要使用for循环
                for node in range(num_nodes):
                    nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
                    a_tmp[node,nei_index] = 1
                    a_tmp[nei_index,node] = 1
                # one-hop indices
                mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops_2[i,m,1:,0])
                A_[i,m,:num_nodes,:num_nodes] = a_tmp      
                feat[i,m,:num_nodes] = feats[i,uni_tmp]
    def change_A(self,B,N,uni,knn_graph,mask_one_hop_idcs,hops_2,A_,feat,feats):
        for i in B:
            for m in range(N):
                uni_tmp = torch.unique(uni[i,m])
                if m not in uni_tmp:
                    uni_tmp = torch.cat((torch.tensor([m]).cuda(non_blocking=True),uni_tmp))
                num_nodes = len(uni_tmp)
                a_tmp = torch.zeros(num_nodes,num_nodes) # 小维度的矩阵cpu上进行索引和改值更快一点
                neighbors = knn_graph[i,uni_tmp,1:self.active_connection+1]
                # 每个结点能够连接的顶点数不同，需要使用for循环
                for node in range(num_nodes):
                    nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
                    a_tmp[node,nei_index] = 1
                    a_tmp[nei_index,node] = 1
                # one-hop indices
                mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops_2[i,m,1:,0])
                A_[i,m,:num_nodes,:num_nodes] = a_tmp      
                feat[i,m,:num_nodes] = feats[i,uni_tmp]
    def __call__(self, feats):
        B,N,D=feats.shape
        
        # ## 1. get the KNN graph
        knn_graph = self.get_KNN(feats,self.distance)
        #knn_graph = knn_graph[:,:, :self.k_at_hop[0] + 1]

        # 添加第一跳
        hops_1 = knn_graph[:,:,:self.k_at_hop[0] + 1]      # B N [1 K1] 这里的1是center point 的索引
        #hops_2 = hops_1.unsqueeze(-1).repeat(1,1,1,self.k_at_hop[1]+1).clone()  #B N [1 K1] [1 K2]
        #hops = hops_1                  # 只有第一跳
        

        # 构建唯一顶点矩阵 构建邻接矩阵 因为每一个lps的顶点数目都不相同，因此需要使用for 循环
        #uni_array = np.empty((B,N),dtype=object)
        # max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        # feat = feats.unsqueeze(-2).repeat(1,1,max_num_nodes,1)
        # feat[:,:,:,:] = 0
        # A_ = feat.clone()[:,:,:,:max_num_nodes]
        # mask_one_hop_idcs = torch.zeros(B,N,max_num_nodes) == 1

        # 构建邻接矩阵和特征矩阵，单跳的纯矩阵实现
        mask = knn_graph.clone()         #B N N
        mask[:,:,:] = 0
        mask = mask.scatter_(2,knn_graph[:,:,1:self.active_connection+1],1) == True
        a = knn_graph[:,:,1:self.active_connection+1].unsqueeze(1).repeat(1,N,1,1)
        a = a[mask].view(B,N,self.active_connection,self.active_connection)
        b = knn_graph[:,:,1:self.active_connection+1].unsqueeze(-1).repeat(1,1,1,self.active_connection)
        c = (b * N) + a

        feat_ = feats.clone().unsqueeze_(1).repeat(1,N,1,1)     # B N N D
        A_ = feat_.clone()[:,:,:,:N].reshape(B,N,-1)          # B N N*N
        A_[:,:,:] = 0
        A_ = A_.scatter_(-1,c.flatten(-2,-1),1).view(B,N,N,N)
        # sparse matrix for feat
        idx = torch.cat((torch.arange(0,B).unsqueeze_(-1).repeat(1,N*self.active_connection).flatten().unsqueeze_(0),torch.arange(0,N).unsqueeze_(-1).unsqueeze_(0).repeat(B,1,self.active_connection).flatten().unsqueeze_(0), knn_graph[:,:,1:self.active_connection+1].flatten().unsqueeze_(0).cpu())).cuda(non_blocking=True)
        feat= torch.sparse_coo_tensor(idx,feat_[mask],(B,N,N,D)).cuda(non_blocking=True) # B N N D

        mask_one_hop_idcs = mask
        #mask_one_hop_idcs = mask_one_hop_idcs.scatter_(2,knn_graph[:,:,1:self.active_connection+1],1) == True

        # 效率问题不考虑第二跳
        # 添加第二跳
        # for i in range(B):
        #     for m in range(N):
        #         hops_2[i,m,:,:] = hops_1[i,hops_1[i,m,:],:self.k_at_hop[1]+1] 

        # hops_1 = hops_1.share_memory_()
        # hops_2 = hops_2.share_memory_()
        # knn_graph = knn_graph.share_memory_()
        # A_ = A_.share_memory_()
        # feat = feat.share_memory_()
        # feats = feats.share_memory_()

        # num_worker = 2
        # processes = []
        # for rank in range(num_worker):
        #     p = mp.Process(target=self.change_hops,args=(range(rank*int(B/num_worker),(rank+1)*int(B/num_worker)),N,hops_1,hops_2,knn_graph,mask_one_hop_idcs,A_,feat,feats))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        #del hops_1
        # hops_2矩阵中，dim -1 第一个元素存储的是K1跳的顶点，后几个元素为K2跳个顶点，dim -2 的第一个元素是中心点的索引
        # 展平hops矩阵后两维，这里就不考虑自身的那一维
        #uni = hops_2[:,:,1:,:].flatten(start_dim=-2, end_dim=-1)
        #uni = hops[:,:,1:]



        # for i in range(B):
        #     for m in range(N):
        #         uni_tmp = torch.unique(uni[i,m])
        #         if m not in uni_tmp:
        #             uni_tmp = torch.cat((torch.tensor([m]).cuda(non_blocking=True),uni_tmp))
        #         num_nodes = len(uni_tmp)
        #         a_tmp = torch.zeros(num_nodes,num_nodes) # 小维度的矩阵cpu上进行索引和改值更快一点
        #         neighbors = knn_graph[i,uni_tmp,1:self.active_connection+1]
        #         # 每个结点能够连接的顶点数不同，需要使用for循环
        #         for node in range(num_nodes):
        #             nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
        #             a_tmp[node,nei_index] = 1
        #             a_tmp[nei_index,node] = 1
        #         # one-hop indices
        #         mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops_2[i,m,1:,0])
        #         A_[i,m,:num_nodes,:num_nodes] = a_tmp      
        #         feat[i,m,:num_nodes] = feats[i,uni_tmp]
        # 正则化邻接矩阵？
        D = A_.sum(-1,keepdim=True)
        A_.div_(D)
        A_[torch.isnan(A_)] = 0
        del D

        # 正则化特征，IPS特征减去中心点特征
        feat = (-feats.unsqueeze(1).repeat(1,N,1,1)).add(feat)
        #feat.sub_(feats.unsqueeze(-2))

        return feat,A_,mask_one_hop_idcs,hops_1[:,:,1:]
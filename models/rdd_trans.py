from math import gamma
from networkx.algorithms import cluster
from numpy.core.fromnumeric import size
import torch.nn as nn
from timm.models.registry import register_model
from .gcn_cluster import *
from .spectral_clustering import *
from .pict_backbone import create_swin_transformer
from .kmeans import kmeans,kmeans_predict

class Attention(nn.Module):
    def __init__(self,classes,in_dim=1536,out_dim=384):
        super(Attention, self).__init__()
        self.L = in_dim
        self.D = out_dim
        self.K = 1
        self.classes = classes
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.GELU(),
            #nn.Tanh(),
            #nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.classes),
            #nn.Sigmoid()
        )
        initialize_weights(self.attention)
        initialize_weights(self.classifier)
    def forward(self, x, bs, return_atte=False):     
        H = x.view(bs,-1,self.L) # B*NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = A@H  # KxL
        M = M.squeeze()
        Y_prob = self.classifier(M)
        if return_atte:
            return Y_prob.view(bs,-1),A
        else:
            return Y_prob.view(bs,-1)

class RddTransformer(nn.Module):
    # 二分类考虑用BCE + Node (512,1)  多分类使用CE + Node(512,cls)
    def __init__(self, backbone=nn.Module,cluster=GCN(),dim=768,**kwargs):
        super().__init__()
        self.cluster_model = cluster
        self.cluster_distance = kwargs['cluster_distance']
        self.nor_index = kwargs['nor_index']
        self.cluster_num = None
        self.cluster_flip_sel = kwargs['cluster_flip_sel']
        if type(cluster)==GCN:
            self.graph = kwargs['graph']
            self.clustre_thr = kwargs['cluster_thr']
        elif cluster == kmeans:
            self.cluster_num = kwargs['num_cluster']
            self.persistent_center = kwargs['persistent_center']
            self.register_parameter('cluster_centers',nn.Parameter(torch.zeros(size=(self.cluster_num,dim)),requires_grad=False))
        
        elif cluster == spectral_clustering:
            self.cluster_num = kwargs['num_cluster']
            self.clustre_rbf_distance = kwargs['cluster_rbf_distance']
            self.cluster_rbf_gamma = kwargs['cluster_rbf_gamma']
            self.n_compoents = kwargs['cluster_n_compoents'] if 'cluster_n_compoents' in kwargs and kwargs['cluster_n_compoents'] is not None else self.cluster_num
            #print()
            self.register_parameter('cluster_centers',nn.Parameter(torch.zeros(size=(self.cluster_num,self.n_compoents)),requires_grad=False))

        self.thr = kwargs.pop('select_cluster_thr')
        num_classes = kwargs.pop('num_classes')
        self.instance_feature_extractor=backbone
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head_instance = nn.Linear(dim, kwargs['ins_num_classes'])  #实例分类器为二分类器，主要用于判断实例是否为病害 0正常 1病害
        self.head = nn.Sequential(
            nn.Linear(dim,num_classes) if num_classes > 0 else nn.Identity()
        )
        self.soft_max = nn.Softmax(-1)
        initialize_weights(self.head_instance)
        initialize_weights(self.head)
    
    def get_cluster_feat_f(self,clusters_feat,cluster_num):
        B = len(clusters_feat)
        feats_tmp = []
        clusters_num = [] if cluster_num == None else [cluster_num] * B
        for b in range(B):
            if cluster_num is not None:
                C = cluster_num
            else:
                C = len(clusters_feat[b])
                clusters_num.append(C)
            
            for i in range(C):
                # score, index_inst = torch.max(scores_inst[b,clusters_idcs[b][i],1], dim=0)
                # if i == 0:
                #     scores = score.unsqueeze_(0)
                # else:
                #     scores = torch.cat((scores,score.unsqueeze_(0)))
                if b==0 and i == 0:
                    feats_tmp = self.avgpool(clusters_feat[b][i].transpose(0,1)).unsqueeze(0)
                else:
                    feats_tmp = torch.cat((feats_tmp,self.avgpool(clusters_feat[b][i].transpose(0,1)).unsqueeze(0)))
            # 选取病害置信度最高实例所在的簇，并将这个簇中所有实例的平均特征作为包特征，对于病害包来说，这是完全正确的，需要选取病害部分来做为包的代表，对于正常包来说，这样做能够使得网络模型的鲁棒性更强，这主要是因为这要求正常包里最像病害的部分也强约束成正常
            #max_clu_index = torch.argmax(scores)
            # if b == 0:
            #     feats = clusters_feat[b][max_clu_index]
            #     feats = self.avgpool(feats.transpose(0, 1)).unsqueeze_(0)  # B C 1
            # else:
            #     feats = torch.cat((feats,self.avgpool(clusters_feat[b][max_clu_index].transpose(0, 1)).unsqueeze_(0)))
        return feats_tmp,clusters_num
    def get_cluster_feat_mask(self,inst_feat,cluster_num,clusters_mask):
        '''
        clusters_mask    C B D
        '''
        B,N,D = inst_feat.shape
        clusters_num = [] if cluster_num == None else [cluster_num] * B
        feats_tmp = []
        if cluster_num is not None:
            C = cluster_num
        else:
            C = len(clusters_mask)
        for b in range(B):
            c = 0
            for i in range(C):
                f_tmp = inst_feat[b][clusters_mask[i][b]]
                if len(f_tmp) != 0:
                    c +=1
                    if len(feats_tmp) == 0:
                        feats_tmp = self.avgpool(f_tmp.transpose(0,1)).unsqueeze(0)
                    else:
                        feats_tmp = torch.cat((feats_tmp,self.avgpool(f_tmp.transpose(0,1)).unsqueeze(0)))  
            if cluster_num is None:
                clusters_num.append(c)
        return feats_tmp,clusters_num
    #使用已经分簇好的特征
    def cluster_classifier(self,clusters_feat,scores_inst,clusters_idcs,thr = 0.8,cluster_num=None,clusters_mask=None):
        B = len(clusters_feat)
        D = clusters_feat[0][0].size(-1)
        if clusters_mask is not None:
            feats_tmp,clusters_num = self.get_cluster_feat_mask(clusters_feat,cluster_num,clusters_mask)
        else:
            feats_tmp,clusters_num = self.get_cluster_feat_f(clusters_feat,cluster_num)
            
        #feats = feats.view(B,-1)
        feats_tmp = feats_tmp.view(-1,D)
        feats_tmp = self.head(feats_tmp)
        scores = self.soft_max(feats_tmp)
        # 认为没有判别性的图块整个类别得分是平坦的，而有判定性的图块，得分呈现凸出状
        if self.nor_index < 0:
            scores,_ = torch.max(scores,dim=-1)
        else:
            scores = 1 - scores[:,self.nor_index]

        try:
            scores = scores.view(B,cluster_num)
        except:
            cluster_num = None
        #簇数量固定时，尽量不使用循环
        if cluster_num is not None:
            mask_max = scores.clone()
            mask_max[:,:] = 0
            max_clu_index = torch.argmax(scores,dim=1).view(B,1)
            mask_max = mask_max.scatter_(1,max_clu_index,1) == 1
            # 在测试阶段，如果最高病害置信度小于一定值，那我认为它是正常包，使用置信度最低的一个簇
            if not self.training and self.nor_index >= 0 and self.cluster_flip_sel:
                mask_min = scores.clone() 
                mask_min[:,:] = 0
                min_clu_index = torch.argmin(scores,dim=1).view(B,1)
                mask_min = mask_min.scatter_(1,min_clu_index,1) == 1
                inverse_mask = scores[mask_max] < thr
                mask_max[inverse_mask] = mask_min[inverse_mask]
            feats_tmp = feats_tmp.view(B,cluster_num,-1)
            feats = feats_tmp[mask_max]
        else:
            j=0
            mask_max = []
            for b in range(B):
                max_clu_index = torch.argmax(scores[j:j+clusters_num[b]])
                # 在测试阶段，如果最高病害置信度小于一定值，那我认为它是正常包，使用置信度最低的一个簇
                if not self.training and scores[j+max_clu_index] < thr and self.nor_index >= 0 and self.cluster_flip_sel:
                    max_clu_index = torch.argmin(scores[j:j+clusters_num[b]])
                if b == 0:
                    feats = feats_tmp[j:j+clusters_num[b]][max_clu_index]
                else:
                    feats = torch.cat((feats,feats_tmp[j:j+clusters_num[b]][max_clu_index]))
                mask_max += [max_clu_index]
                j = j+clusters_num[b]

        return feats.view(B,-1),clusters_num,mask_max,scores
    
    # for sklear-based cluster api，主要补充了循环和返回的mask
    def sklearn_cluster(self,inst_feature):
        #assert len(instance_feat.shape) == 3 and len(clusters_indic.shape)==2
        B,N,D = inst_feature.shape
        # 对kmeans使用循环，这是由于算法本质导致的，无法实现为batch-based
        if self.cluster_model == kmeans:
            for b in range(B):
                if self.training:
                    # 这里更改原始实现里最后返回部分，原始实现中这里最后强制返回cpu向量，我去掉了这一部分
                    clu_labels,self.get_parameter('cluster_centers').data = self.cluster_model(X=inst_feature[b],num_clusters=self.cluster_num,device=torch.cuda.current_device(),cluster_centers = self.get_parameter('cluster_centers').data if self.get_parameter('cluster_centers').data.any() != 0 and self.persistent_center else [],tqdm_flag=False,distance=self.cluster_distance)
                else:
                    if self.persistent_center:
                        clu_labels = kmeans_predict(X=inst_feature[b],device=torch.cuda.current_device(),cluster_centers = self.get_parameter('cluster_centers').data,tqdm_flag=False,distance=self.cluster_distance)
                    else:
                        clu_labels,_ = self.cluster_model(X=inst_feature[b],num_clusters=self.cluster_num,device=torch.cuda.current_device(),tqdm_flag=False,distance=self.cluster_distance)
                clu_labels = torch.unsqueeze(clu_labels,dim=0)
                if b == 0:
                        clusters_idcs = clu_labels.clone()
                else:
                    clusters_idcs = torch.cat((clusters_idcs,clu_labels))
        elif self.cluster_model == spectral_clustering:
            output = self.cluster_model(feats=inst_feature,n_clusters=self.cluster_num,
            cluster_centers = self.get_parameter('cluster_centers').data if self.get_parameter('cluster_centers').data.any() != 0 else [],
            kmeans_distance=self.cluster_distance,
            rbf_distance=self.clustre_rbf_distance,
            gamma = self.cluster_rbf_gamma,
            n_components = self.n_compoents,
            is_training=self.training)

            if isinstance(output, (tuple, list)):
                [clusters_idcs,self.get_parameter('cluster_centers').data] = output
            else:
                clusters_idcs = output

        #cluster mask for generate cluster features
        for i in range(self.cluster_num):
            cluster_mask = clusters_idcs - i == 0
            if i == 0:
                clusters_mask = cluster_mask.clone().unsqueeze(0)
            else:
                clusters_mask = torch.cat((clusters_mask,cluster_mask.clone().unsqueeze(0)))
        return clusters_idcs,clusters_mask


    def forward(self,x,is_teacher=False,return_inst=False):
        # step 1, get the instance feat by backbone Network
        avg_bag_feature, inst_feature=self.instance_feature_extractor.forward_features(x) #B*N*D
        B,N,D = inst_feature.shape
        # step 2, cluster 
        if type(self.cluster_model) == GCN:
            # if using gcn to cluster, firstly create the graph
            feat, adj, h1_mask,h1_indi = self.graph(inst_feature,inst_feature.is_cuda)
            
            #adj = self.graph.get_KNN_adj(feat=inst_feature)
            #print(adj.size())
            torch.cuda.empty_cache()
            # gcn cluster  edges, scores
            
            if is_teacher:
                logits_edge = self.cluster_model(feat, adj, h1_mask)
                return logits_edge,h1_mask
            else:
                logits_edge = self.cluster_model(feat, adj, h1_mask)
            pred = self.soft_max(logits_edge)
            del feat, adj
            torch.cuda.empty_cache()
            #clusters_feat,clusters_idcs = gcn_cluster(None,adj, inst_feature,self.clustre_thr) # C*N*D
            clusters_feat,clusters_idcs = gcn_cluster(h1_indi,pred, inst_feature,self.clustre_thr,inst_feature.is_cuda)
            cluster_num = None
            clusters_mask = None
        elif self.cluster_model == spectral_clustering:
            # spectral_cluster
            cluster_num = self.cluster_num
            # find cluster features
            clusters_idcs,clusters_mask = self.sklearn_cluster(inst_feature)
            clusters_feat = inst_feature
            # 谱聚类由于降维过多，导致在batch-based kmeans 训练中可能出现实际聚类数量少于设定值的情况
            if torch.max(clusters_idcs,dim=1)[0].any() < self.cluster_num-1:
                cluster_num = None
        elif self.cluster_model == kmeans:
            # kmeans cluster 
            cluster_num = self.cluster_num
            # find cluster features
            clusters_idcs,clusters_mask = self.sklearn_cluster(inst_feature)
            clusters_feat = inst_feature
        elif type(self.cluster_model) == Attention:
            self.cluster_model()
        # 分簇包分类
        if self.cluster_model is not None:
            #测试时聚类方法都认为不存在固定聚类数目
            if not self.training:
                cluster_num = None

        #测试时聚类方法都认为不存在固定聚类数目
        if not self.training:
            cluster_num = None
        # for i in range(len(clusters_feat)):
        #     for m in range(len(clusters_feat[i])):
        #         print(clusters_feat[i][m].size())
        #         print(clusters_idcs[i][m])
        # B C N* D

        # step 3 classify
        # instance classify
        #if self.training or return_inst:
        logits_inst = self.head_instance(inst_feature.view(-1,D))
        logits_inst = logits_inst.view(B,N,-1)
        #score_inst = self.soft_max(logits_inst)
        # bag classify
        if self.cluster_model is not None:
            logits_bag,clusters_num,_,_= self.cluster_classifier(clusters_feat,None,clusters_idcs,thr=self.thr,cluster_num = cluster_num,clusters_mask=clusters_mask)
        else:
            logits_bag,clusters_num = self.head(avg_bag_feature),[1]
        
        # except:
        #     np.savez('/mnt/d/wsl/output/test.npz',mask=clusters_mask.cpu().numpy(),idcs=clusters_idcs.cpu().numpy())
        if type(self.cluster_model) == GCN:
            return logits_bag, logits_inst,logits_edge,h1_mask, clusters_num
        if self.training or return_inst:
            #return logits_bag, logits_inst, score_inst,clusters_num
            return logits_bag, logits_inst,clusters_num
            #return logits_bag, None, None,clusters_num
        else:
            # bag classify
            #return logits_bag, None, None,clusters_num
            return logits_bag, logits_inst,clusters_num

@register_model
def rdd_trans_swin_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    backbone = create_swin_transformer('swin_small_patch4_window7_224', pretrained=pretrained, **model_kwargs)
    if 'cluster_name' in kwargs:
        if kwargs['cluster_name'].lower() == 'kmeans':
            return RddTransformer(backbone=backbone,cluster=kmeans,**kwargs)
        elif kwargs['cluster_name'].lower() == 'gcn':
            return RddTransformer(backbone=backbone,cluster=GCN(in_dim=768,out_dim=384,k1=kwargs['ips_k_at_hop'][0]),graph = KnnGraph(kwargs['ips_active_connection'],kwargs['ips_k_at_hop'],kwargs['cluster_distance']),**kwargs)
        elif kwargs['cluster_name'].lower() == 'spectral':
            return RddTransformer(backbone=backbone,cluster=spectral_clustering,**kwargs)
        elif kwargs['cluster_name'].lower() == 'attention':
            return RddTransformer(backbone=backbone,cluster=Attention(classes=kwargs['num_classes'],in_dim=768,out_dim=192),**kwargs)
        else:
            return RddTransformer(backbone=backbone,cluster=None,**kwargs)
    else:
        return RddTransformer(backbone=backbone,cluster=None,**kwargs)

@register_model
def rdd_trans_swin_base_patch4_window12_384_in22k(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),**kwargs)
    backbone = create_swin_transformer('swin_base_patch4_window12_384_in22k', pretrained=pretrained, **model_kwargs)
    if 'cluster_name' in kwargs:
        if kwargs['cluster_name'].lower() == 'kmeans':
            return RddTransformer(backbone=backbone,cluster=kmeans,dim=1024,**kwargs)
        elif kwargs['cluster_name'].lower() == 'gcn':
            return RddTransformer(backbone=backbone,cluster=GCN(in_dim=1024,out_dim=256,k1=kwargs['ips_k_at_hop'][0]),graph = KnnGraph(kwargs['ips_active_connection'],kwargs['ips_k_at_hop'],kwargs['cluster_distance']),dim=1024,**kwargs)
        elif kwargs['cluster_name'].lower() == 'spectral':
            return RddTransformer(backbone=backbone,cluster=spectral_clustering,dim=1024,**kwargs)
        elif kwargs['cluster_name'].lower() == 'attention':
            return RddTransformer(backbone=backbone,cluster=Attention(classes=kwargs['num_classes'],in_dim=1024,out_dim=256),dim=1024,**kwargs)
        else:
            return RddTransformer(backbone=backbone,cluster=None,dim=1024,**kwargs)
    else:
        return RddTransformer(backbone=backbone,cluster=None,dim=1024,**kwargs)

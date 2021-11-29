import torch.nn as nn
from timm.models.registry import register_model
from .gcn_cluster import *
from .swin import _create_swin_transformer
from cv2 import kmeans

class RddTransformer(nn.Module):
    # 二分类考虑用BCE + Node (512,1)  多分类使用CE + Node(512,cls)
    def __init__(self, backbone=nn.Module,cluster=GCN(), graph=KnnGraph(),dim=768,**kwargs):
        super().__init__()
        self.cluster_model = cluster
        self.cluster_distance = kwargs['cluster_distance']
        if type(cluster)==GCN:
            self.graph = graph
            self.clustre_thr = kwargs['cluster_thr']
        elif cluster == kmeans:
            self.cluster_centers = []
            self.cluster_num = kwargs['num_cluster']
        num_classes = kwargs.pop('num_classes')
        self.instance_feature_extractor=backbone
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head_instance = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(dim,num_classes)
        )
        initialize_weights(self.head_instance)
        initialize_weights(self.head)

    def cluster_classifier(self,clusters_feat,scores_inst,clusters_idcs):
        B = len(clusters_feat)
        for b in range(B):
            result={'clusters_score':[],'score':[],'index':[]}
            for i in range(len(clusters_feat[b])):
                print(clusters_idcs[b][i])
                print(scores_inst.size())
                score, index = torch.max(scores_inst[b,clusters_idcs[b][i]], dim=0)
                result['score'].append(score)
                result['index'].append(index)
            scores = torch.stack(result['score'],1)
            max_value, index = torch.max(scores, dim=1)
            feats = clusters_feat[b][index]
            feats = self.avgpool(feats.transpose(1, 2))  # B C 1
            logits = self.head(feats)
        return logits
    
    # for kmeans
    def kmeans_cluster(self,instance_feat,clusters_indic):
        #assert len(instance_feat.shape) == 3 and len(clusters_indic.shape)==2
        print(clusters_indic.size())
        clusters_feat = list()
        for i in range(self.cluster_num):
            cluster_indic = clusters_indic - i == 0
            print(cluster_indic)
            print(instance_feat[cluster_indic].size())
            clusters_feat.append(instance_feat[cluster_indic])
        return clusters_feat

    def forward(self,x,bag_label=None,is_training=False):
        # step 1, get the instance feat by backbone Network
        _, inst_feature=self.instance_feature_extractor.forward_features(x) #B*N*D
        
        B,N,D = inst_feature.shape
        # step 2, cluster 
        if type(self.cluster_model) == GCN:
            # if using gcn to cluster, firstly create the graph
            feat, adj, h1_mask,h1_indi = self.graph(inst_feature)
            # gcn cluster  edges, scores
            pred = self.cluster_model(feat, adj, h1_mask)
            clusters_feat,clusters_idcs = gcn_cluster(h1_indi,pred.view(B,N,-1,2), inst_feature,self.clustre_thr) # C*N*D
        # 暂时放弃kmeans
        else:
            # kmeans cluster 
            for b in range(B):
                clu_labels,self.cluster_centers = self.cluster_model(X=inst_feature[b],num_clusters=self.cluster_num,device=torch.cuda.current_device(),cluster_centers = self.cluster_centers,tqdm_flag=False,distance=self.cluster_distance)
                clu_labels = torch.unsqueeze(clu_labels,dim=0)
                if b == 0:
                     cluster_indic = clu_labels.clone()
                else:
                    cluster_indic = torch.cat((cluster_indic,clu_labels))
            # find cluster features
            clusters_feat = self.kmeans_cluster(inst_feature,cluster_indic)
        # for i in range(len(clusters_feat)):
        #     for m in range(len(clusters_feat[i])):
        #         print(clusters_feat[i][m].size())
        # B C N* D
        # step 3 classify
        # instance classify
        logits_inst = self.head_instance(inst_feature.view(-1,D))
        logits_inst = logits_inst.view(B,N,-1)
        # bag classify
        logits_bag = self.cluster_classifier(clusters_feat,logits_inst,clusters_idcs)
        if is_training:
            return logits_bag, logits_inst
        else:
            return logits_bag, logits_inst

@register_model
def cluster_swin_small_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    backbone = _create_swin_transformer('swin_small_patch4_window7_224', pretrained=pretrained, **model_kwargs)
    if kwargs['cluster_name'].lower() == 'kmeans':
        return RddTransformer(backbone=backbone,cluster=kmeans,**kwargs)
    elif kwargs['cluster_name'].lower() == 'gcn':
        return RddTransformer(backbone=backbone,cluster=GCN(in_dim=768,out_dim=384,k1=kwargs['ips_k_at_hop'][0]),graph = KnnGraph(kwargs['ips_active_connection'],kwargs['ips_k_at_hop'],kwargs['cluster_distance']),**kwargs)
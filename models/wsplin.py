import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.models.registry import register_model


class FeatureExtractorNetwork(nn.Module):
    def __init__(self,backbone,attention=False):
        super(FeatureExtractorNetwork, self).__init__()

        self.attention = attention
        self.model = backbone

    def forward(self, x):
        # input shape is [batch_size, patches, 3, 300, 300]
        bs = x.size(0)
        x = x.flatten(0,1)
        # x = x.view(-1, 3, 300, 300)
        if self.attention:
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            x = self.model.forward_features(x)
            x = self._avg_pooling(x)
        else:
            x = self.model(x)
            # output shape is [batch_size*patches, NUM_CLASSES]
            # reshape it to [batch_size, patches*NUM_CLASSES]
            x = x.view(bs, -1)
        return x


class Attention(nn.Module):
    def __init__(self,classes):
        super(Attention, self).__init__()
        self.L = 1536
        self.D = 384
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

    def forward(self, x, bs):     
        H = x.view(bs,-1,self.L) # B*NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = A@H  # KxL
        M = M.squeeze()
        Y_prob = self.classifier(M)
        
        return Y_prob.view(bs,-1)

class ClassifierNetwork(nn.Module):
    def __init__(self, num_classes,patches,dp_rate=0.5):
        super().__init__()

        self.cls_head = nn.Sequential(
            nn.Linear(num_classes*patches, num_classes*patches),
            nn.ReLU(),
            nn.Dropout(p=dp_rate),
            nn.Linear(num_classes*patches, num_classes)
        )
        
    def forward(self, x,bs=0):

        return self.cls_head(x)


class WSPLIN_IP(nn.Module):
    def __init__(self,backbone,num_classes=8,attention=False,patches=17,dp_rate=0.5):
        super().__init__()
        self.feature_extractor = FeatureExtractorNetwork(backbone,attention)
        self.classifier = Attention(num_classes) if attention else ClassifierNetwork(num_classes,patches,dp_rate)
    def forward(self, x,bs=0):
        x = self.feature_extractor(x)
        x = self.classifier(x,bs)
        return x

@register_model
def wsplin_effi_b3(pretrained=False,**kwargs):
    config = kwargs.pop('config')
    if config.WSPLIN.BACKBONE_INIT is not None:
        pretrained = False
    img_size = kwargs.pop('img_size')
    backbone = create_model(
        model_name='tf_efficientnet_b3',
        pretrained=pretrained,
        **kwargs
    )
    if config.WSPLIN.BACKBONE_INIT is not None:
        cpt = torch.load(config.BACKBONE_INIT, map_location = 'cpu')
        backbone.load_state_dict(cpt,strict=False)
    return WSPLIN_IP(backbone,num_classes=config.MODEL.NUM_CLASSES,attention=config.WSPLIN.ATTENTION,patches=math.ceil(config.WSPLIN.NUM_PATCHES * config.WSPLIN.SPARSE_RATIO),dp_rate=config.WSPLIN.CLS_HEAD_DP_RATE)

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import create_model
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
class STN(nn.Module):
    def __init__(self, backbone,type=1):
        super().__init__()

        self.backbone = backbone
        self._type = type
        # Spatial transformer localization-network 
        # ref pytorch tutorial
        # https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        # # Spatial transformer localization-network
        if type == 2:
            self.localization = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 71 * 71, 32),
                nn.ReLU(True),
                nn.Linear(32, 3 * 2)
            )
            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        # imple for https://arxiv.org/abs/2203.09580
        elif type == 0:
            self.localization = nn.Sequential(
                nn.Conv2d(3, 100, kernel_size=7),
                #nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),

                nn.Conv2d(100, 100, kernel_size=5),
                #nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),

                nn.Conv2d(100, 50, kernel_size=3),
                nn.ReLU(True),
                # nn.MaxPool2d(2, stride=2),
            )
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Conv2d(50,6,kernel_size=1),
                nn.AdaptiveAvgPool2d(1)
            )
        # imple for https://arxiv.org/abs/2203.09580, add the BN layer
        elif type == 1:
            self.localization = nn.Sequential(
                nn.Conv2d(3, 100, kernel_size=7),
                #nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(100),
                nn.ReLU(True),

                nn.Conv2d(100, 100, kernel_size=5),
                #nn.MaxPool2d(2, stride=2),
                nn.BatchNorm2d(100),
                nn.ReLU(True),

                nn.Conv2d(100, 50, kernel_size=3),
                nn.BatchNorm2d(50),
                nn.ReLU(True),
                # nn.MaxPool2d(2, stride=2),
            )
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Conv2d(50,6,kernel_size=1),
                nn.AdaptiveAvgPool2d(1)
            )

        # imple for https://github.com/aicaffeinelife/Pytorch-STN/blob/master/models/STNModule.py
        elif type == 3:
            self.localization = nn.Sequential(
                nn.Conv2d(3, 100, kernel_size=7,bias=False,padding='same'),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(100, 100, kernel_size=5,bias=False,padding='same'),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(100, 32, kernel_size=3,bias=False,padding='same'),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
            )
            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(32*37*37,1024),
                nn.Linear(1024,6)
            )
        self.apply(self._init_weights)
    def stn(self,x):
        bs = x.size(0)
        xs = self.localization(x)
        # print(xs.size())
        if self._type == 2:
            xs = xs.view(-1, 10 * 71 * 71)
            # 300,300,50
            # print(xs.size())
            xs = xs.view(bs,-1)
        elif self._type == 3:
            xs = xs.view(-1, 32*37*37)
            # 300,300,50
            # print(xs.size())
            xs = xs.view(bs,-1)
        theta = self.fc_loc(xs)
        # 1,1,6
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x,grid

    def forward_features(self,x):
        x,_ = self.stn(x)
        return self.backbone.forward_features(x)

    def forward(self, x):
        x,_ = self.stn(x)
        return self.backbone(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.backbone, 'no_weight_decay'):
            return {'backbone.' + i for i in self.backbone.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.backbone, 'no_weight_decay_keywords'):
            return {'backbone.' + i for i in self.backbone.no_weight_decay_keywords()}
        return {}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

@register_model
def stn_1(pretrained=False,backbone=None,**kwargs):
    config = kwargs.pop('config')
    backbone = backbone

    return STN(backbone,0)

@register_model
def stn_1_bn(pretrained=False,backbone=None,**kwargs):
    config = kwargs.pop('config')
    backbone = backbone

    return STN(backbone,1)

@register_model
def stn_2(pretrained=False,backbone=None,**kwargs):
    config = kwargs.pop('config')
    backbone = backbone

    return STN(backbone,2)

@register_model
def stn_3(pretrained=False,backbone=None,**kwargs):
    config = kwargs.pop('config')
    backbone = backbone

    return STN(backbone,3)
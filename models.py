import torch
import torch.nn as nn
from timm.models import create_model
import swin
import deit
MODEL_NAME = 'efficientnet-b3'

class FeatureExtractorNetwork(nn.Module):
    def __init__(self,is_pretrain,arch,shard_id,num_classes,patch_size,path):
        super(FeatureExtractorNetwork, self).__init__()
        # use fine tuned b3
        self.arch=arch
        self.patch_size = patch_size
        if arch=="effi-b3":
            original_model = EfficientNet.from_name(MODEL_NAME,num_classes=num_classes)
            checkpoint = torch.load(path, map_location='cpu')
            original_model.load_state_dict(checkpoint['state_dict'])
            #original_model.load_state_dict(torch.load(PARA_PATH))
        else:
            if arch=="effi-b3":
                original_model = EfficientNet.from_pretrained(MODEL_NAME,num_classes=num_classes)
        self.model = original_model

    def forward(self, x):
        # input shape is [batch_size, config.MODEL.NUM_PATCHES, 3, 300, 300]
        bs = x.size(0)
       # if self.arch=='effi-b3':
           #x = x.view(-1, 3, 300, 300)
        x = x.view(-1, 3, self.patch_size, self.patch_size)
        x = self.model(x)
        # output shape is [batch_size*config.MODEL.NUM_PATCHES, NUM_CLASSES]
        # reshape it to [batch_size, config.MODEL.NUM_PATCHES*NUM_CLASSES]
        x = x.view(bs, -1)
        #x = self.model.extract_features(x)
        return x


class ClassifierNetwork(nn.Module):
    def __init__(self, num_patches,num_classes,drop_rate):
        super().__init__()
        # 119 = 7*17
        # 1536 is output features of b3
        '''self.dense1 = nn.Linear(num_classes*config.MODEL.NUM_PATCHES, 119)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(num_classes*config.MODEL.NUM_PATCHES, num_classes)'''
        '''self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(1536*num_patches, 1536)
        self.dense2 = nn.Linear(1536, num_classes)
        self.dense3 = nn.Linear(1536*num_patches, num_classes)'''
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.dense4 = nn.Linear(num_classes*num_patches, num_classes*num_patches)
        self.dense5 = nn.Linear(num_classes*num_patches, num_classes)
        
    def forward(self, x):
        '''x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.dense2(x)'''
        
        '''bs = int(x.size(0)/config.MODEL.NUM_PATCHES)
        x = self._avg_pooling(x)
        x = x.view(bs,-1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)'''
        x = self.dense4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense5(x)
        #x = self.dense3(x)
        return x

class WSPLIN_IP(nn.Module):
    def __init__(self,shard_id,is_pretrain=True,arch='effi-b3',num_classes=8,num_patches=17,drop_rate=0.5,patch_size=300,path=''):
        super().__init__()
        self.feature_extractor = FeatureExtractorNetwork(is_pretrain,arch,shard_id,num_classes,patch_size,path=path)
        self.classifier = ClassifierNetwork(num_classes=num_classes,num_patches=num_patches,drop_rate=drop_rate)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def build_model(config):
    model_name=config.MODEL.NAME
    if model_name=='wsplin':
        model = WSPLIN_IP(config.LOCAL_RANK,
                          num_classes=config.MODEL.NUM_CLASSES,
                          num_patches=config.MODEL.NUM_PATCHES,
                          drop_rate=config.MODEL.DROP_RATE,
                          patch_size=config.DATA.PATCH_SIZE,
                          path=config.DATA.PRETRAINED_DIR)
    # Use the official impl to use the gradient cheackpoint
    #elif model_name.startswith('swin'):
        #model = build_swin_model(config)
    else:
        model = create_model(
            config.MODEL.NAME,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
    return model
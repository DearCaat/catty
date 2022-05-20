### reference https://github.com/TACJu/TransFG
### change the ViT input size from 224 to any

from timm.models.registry import register_model
from timm.models import create_model
import torch
from torch.hub import load_state_dict_from_url


def load_model_weights(model, model_path):
    ### reference https://github.com/TACJu/TransFG
    ### thanks a lot.
    state = load_state_dict_from_url(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state:
            ip = state[key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print('could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return 

@register_model
def vit_base_patch16_448_miil_in21k(pretrained=False, **kwargs):
    """ Vit-B @ 448x448, trained ImageNet-21k
    """
    model = create_model(
            'vit_base_patch16_224_miil_in21k',
            pretrained=False,
            img_size = 448,
            **kwargs
        )

    img_size = 448
    pretrained_dir = 'https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth'
    if pretrained:
        load_model_weights(model,pretrained_dir)

    ### Vit model input can transform 224 to another, we use linear
    ### thanks: https://github.com/TACJu/TransFG/blob/master/models/modeling.py
    ### similar to https://github.com/rwightman/pytorch-image-models/blob/20a1fa63f8ea999dab29d927d5e1866ed3b67348/timm/models/vision_transformer.py#L588 func resize_pos_embed
    import math
    from scipy import ndimage

    posemb_tok, posemb_grid = model.pos_embed[:, :1], model.pos_embed[0, 1:]
    posemb_grid = posemb_grid.detach().numpy()
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = img_size//16
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    model.pos_embed = torch.nn.Parameter(posemb)

    return model
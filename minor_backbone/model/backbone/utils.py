from . import res2net, resnet
import torch.nn as nn

def model_downloader(backbone:str) -> nn.Module:
    print(f'Backbone: {backbone}')
    if backbone.__contains__('resnet'):
        return getattr(resnet, backbone)
    elif backbone.__contains__('res2net'):
        return getattr(res2net, backbone)
    else:
        return None
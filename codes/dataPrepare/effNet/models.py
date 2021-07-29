import torch
import torch.nn as nn
from torchvision import models
from .effnet.model import EfficientNet

class efnet(nn.Module):
    def __init__(self, n_class, modeltype = 0, pretrain=True):
        super().__init__()
        modelname = {0:'efficientnet-b0',1:'efficientnet-b1',2:'efficientnet-b2',3:'efficientnet-b3'}
        if pretrain:
            self.efmodel = EfficientNet.from_pretrained(modelname[modeltype])
        else:
            self.efmodel = EfficientNet.from_name(modelname[modeltype])

        self.feature = self.efmodel._fc.in_features

        self.efmodel._fc = nn.Sequential(
            nn.Linear(in_features=self.feature,out_features=256,bias=True),
            nn.Linear(in_features=256,out_features=64,bias=True),
            nn.Linear(in_features=64,out_features=n_class,bias=True)
        )

    def forward(self, input):
        out = self.efmodel(input)

        return out



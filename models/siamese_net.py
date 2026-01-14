import torch
import torch.nn as nn
from torchvision import models

class SiameseIrisNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x)
        return x

    def forward(self, img1, img2):
        out1 = self.forward_once(img1)
        out2 = self.forward_once(img2)
        return out1, out2

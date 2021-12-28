from typing import List, Dict
from torch import Tensor
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class EarlyStopping():

    def __init__(self, patience = 3, min_delta = 0.001):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.stop = False
        self.wait = 0

    def __call__(self, current_loss):
        
        loss = -current_loss

        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss + self.min_delta:
            self.wait += 1
            print(f'EarlyStopping counter: {self.wait} out of {self.patience}')
            if self.wait >= self.patience:
                self.stop = True
        else:
            self.best_loss = loss
            self.counter = 0

class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])

class ResnetMultiTask(nn.Module):
    """
    Multi-task art classifier.
    (image) -> (resnet50) -> (linear-artist), (linear-style), (linear-genre)
    """

    def __init__(self, num_classes: Dict[str, int], freeze = False):
        super(ResnetMultiTask, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
                                    nn.Dropout(0.5),
                                    layer4
                                    )
        self.resnet.fc = nn.Identity()

        self.class_style = nn.Linear(len_last, num_classes['style'])
        self.class_genre = nn.Linear(len_last, num_classes['genre'])

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_emb = self.resnet(img)
        out_style = self.class_style(visual_emb)
        out_genre = self.class_genre(visual_emb)

        return [out_style, out_genre]

class ResnetSingleTask(nn.Module):
    """
    Single-task art classifier.
    (image) -> (resnet50) -> (linear)
    """

    def __init__(self, num_class:int, freeze = False):
        super(ResnetSingleTask, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        layer4 = self.resnet.layer4
        self.resnet.layer4 = nn.Sequential(
                                    nn.Dropout(0.5),
                                    layer4
                                    )
        self.resnet.fc = nn.Linear(len_last, num_class)

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

            for param in self.resnet.layer4.parameters():
                param.requires_grad = True

            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, img: Tensor) -> Tensor:

        out = self.resnet(img)

        return out
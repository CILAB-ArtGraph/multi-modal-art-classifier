from typing import List, Dict
from torch import Tensor
import torch.nn as nn
from torchvision import models

class EarlyStopping():

    def __init__(self, patience = 3, min_delta = 0.001):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.stop = False
        self.wait = 0

    def __call__(self, current_loss):

        if self.best_loss == None:
            self.best_loss = current_loss
        elif abs(current_loss - self.best_loss) > self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait = self.wait + 1
            print(f"INFO: Early stopping counter {self.wait} of {self.patience}")
            if self.wait >= self.patience:
                self.stop = True

class ResnetMultiTask(nn.Module):
    """
    Multi-task art classifier.
    (image) -> (resnet50) -> (linear-artist), (linear-style), (linear-genre)
    """

    def __init__(self, num_classes: Dict[str, int], freeze = False):
        super(ResnetMultiTask, self).__init__()

        resnet = models.resnet50(pretrained=True)
        len_last = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.class_style = nn.Sequential(nn.Linear(len_last, num_classes['style']))
        self.class_genre = nn.Sequential(nn.Linear(len_last, num_classes['genre']))

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
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

        resnet = models.resnet50(pretrained=True)
        len_last = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(nn.Linear(len_last, num_class))

    def forward(self, img: Tensor) -> Tensor:

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out = self.classifier(visual_emb)

        return out
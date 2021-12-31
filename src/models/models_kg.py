from typing import List, Dict
from torch import Tensor
import torch.nn as nn
from torchvision import models

class MultiModalSingleTask(nn.Module):

    def __init__(self, emb_size: int, num_class:int):
        super(MultiModalSingleTask, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.classifier = nn.Linear(len_last, num_class)

        self.encoder = nn.Linear(len_last, emb_size)


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        out = self.classifier(visual_features)
        graph_proj = self.encoder(visual_features)

        return out, graph_proj

class MultiModalMultiTask(nn.Module):

    def __init__(self, emb_size: int, num_classes: Dict[str, int]):
        super(MultiModalMultiTask, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.class_style = nn.Linear(len_last, num_classes['style'])
        self.class_genre = nn.Linear(len_last, num_classes['genre'])

        self.encoder = nn.Linear(len_last, emb_size)


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)
        graph_proj = self.encoder(visual_features)

        out_style = self.class_style(visual_features)
        out_genre = self.class_genre(visual_features)
        outs = [out_style, out_genre]

        return outs, graph_proj
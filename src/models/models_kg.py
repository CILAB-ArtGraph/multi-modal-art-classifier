from typing import List, Dict
from torch import Tensor
import torch.nn as nn
from torchvision import models


class MultiModalSingleTask(nn.Module):

    def __init__(self, emb_size: int, num_class:int):
        super(MultiModalSingleTask, self).__init__()

        resnet = models.resnet50(pretrained=True)
        len_last = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(nn.Linear(len_last, num_class))

        self.encoder = nn.Sequential(nn.Linear(emb_size, emb_size), nn.Tanh())


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        pred_class = self.classifier(visual_emb)
        graph_proj = self.encoder(visual_emb)

        return [pred_class, graph_proj]

class MultiModalMultiTask(nn.Module):

    def __init__(self, emb_size: int, num_classes: Dict[str, int]):
        super(MultiModalMultiTask, self).__init__()

        resnet = models.resnet50(pretrained=True)
        len_last = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.class_style = nn.Sequential(nn.Linear(len_last, num_classes['style']))
        self.class_genre = nn.Sequential(nn.Linear(len_last, num_classes['genre']))

        self.encoder = nn.Sequential(nn.Linear(emb_size, emb_size), nn.Tanh())


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        graph_proj = self.encoder(visual_emb)

        out_style = self.class_style(visual_emb)
        out_genre = self.class_genre(visual_emb)
        outs = [out_style, out_genre]

        return [outs, graph_proj]
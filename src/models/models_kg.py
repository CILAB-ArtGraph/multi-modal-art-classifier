from typing import List, Dict
from torch import Tensor, cat
import torch.nn as nn
from torchvision import models

class ContextNetSingleTask(nn.Module):

    def __init__(self, emb_size: int, num_class:int):
        super().__init__()

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

class ContextNetlMultiTask(nn.Module):

    def __init__(self, emb_size: int, num_classes: Dict[str, int]):
        super().__init__()

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

class MultiModalSingleTask(nn.Module):

    def __init__(self, emb_size: int, num_class:int):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(len_last + emb_size, num_class)
        )

        self.encoder = nn.Sequential(
            nn.Linear(len_last, emb_size),
            nn.Tanh(),
            nn.Linear(emb_size, emb_size),
            nn.Tanh()
        )

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        graph_proj = self.encoder(visual_features)

        concat = cat((visual_features, graph_proj), 1)

        out = self.classifier(concat)

        return out, graph_proj

class NewMultiModalSingleTask(nn.Module):

    def __init__(self, emb_size: int, num_class:int, dropout: float):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last + emb_size, num_class))


    def forward(self, img: Tensor, embedding: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        comb = cat((visual_features, embedding), dim = 1)

        out = self.classifier(comb)

        return out

class LabelProjector(nn.Module):

    def __init__(self, emb_size: int):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.encoder = nn.Linear(len_last, emb_size)


    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        out = self.encoder(visual_features)

        return out
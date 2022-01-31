from typing import List, Dict
import torch
from torch import Tensor
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from timm import create_model

class EarlyStopping():

    def __init__(self, patience = 3, min_delta = 0.001, checkpoint_path = 'checkpoint.pt'):

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.stop = False
        self.wait = 0
        self.path = checkpoint_path

    def __call__(self, current_loss, model):
        
        loss = -current_loss

        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss < self.best_loss + self.min_delta:
            self.wait += 1
            print(f'EarlyStopping counter: {self.wait} out of {self.patience}')
            if self.wait >= self.patience:
                self.stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        print(f'Validation loss decreased. Saving model...')
        torch.save(model.state_dict(), self.path)

class ResnetMultiTask(nn.Module):
    """
    Multi-task art classifier.
    (image) -> (resnet50) -> (linear-style)
    (image) -> (resnet50) -> (linear-genre)
    """

    def __init__(self, num_classes: Dict[str, int]):
        super(ResnetMultiTask, self).__init__()

        self.resnet = models.resnet50(pretrained = True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.style_classifier = nn.Linear(len_last, num_classes['style'])
        self.genre_classifier = nn.Linear(len_last, num_classes['genre'])

    def forward(self, img: Tensor) -> List[Tensor]:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        out_style = self.style_classifier(visual_features)
        out_genre = self.genre_classifier(visual_features)

        return [out_style, out_genre]

class ResnetSingleTask(nn.Module):
    """
    Single-task art classifier.
    (image) -> (resnet50) -> (linear)
    """

    def __init__(self, num_class: int):
        super(ResnetSingleTask, self).__init__()

        self.resnet = models.resnet50(pretrained = True)
        len_last = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.classifier = nn.Linear(len_last, num_class)

    def forward(self, img: Tensor) -> Tensor:

        visual_features = self.resnet(img)
        visual_features = visual_features.view(visual_features.size(0), -1)

        out = self.classifier(visual_features)

        return out

class ViTSingleTask(nn.Module):

    def __init__(self, num_class: int):
        super(ViTSingleTask, self).__init__()

        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        len_last = self.vit.head.in_features
        self.vit = nn.Sequential(*list(self.vit.children())[:-1])

        self.classifier = nn.Linear(len_last, num_class)

    def forward(self, img: Tensor) -> Tensor:

        #print(img.squeeze().shape)
        visual_features = self.vit(img.squeeze())
        #visual_features = visual_features.view(visual_features.size(0), -1)

        out = self.classifier(visual_features)

        return out
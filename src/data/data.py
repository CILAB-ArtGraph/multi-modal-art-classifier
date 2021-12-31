import os
import torch.utils.data as data
from pandas import DataFrame
from PIL import Image, ImageFile
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from torch import Tensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def transform(image: Tensor):
    preprocess = Compose([
        Resize((224,224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(image)

def vit_transform(image: Tensor):
    preprocess = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return preprocess(image)

class Artgraph(data.Dataset):
    def __init__(self, image_dir:str, df_image_label: DataFrame, transform_type = 'resnet'):
        self.image_dir = image_dir
        self.dataset = df_image_label
        self.transform_type = transform_type

    def __len__(self) -> int:
        return len(self.dataset)

    def prepare_image(self, image_path:str) -> Tensor:

        image = Image.open(image_path)
        if(image.mode != 'RGB'):
            image = image.convert('RGB')
        if self.transform_type == 'resnet':
            image_tensor = transform(image)
        else:
            image_tensor = vit_transform(image)
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor



class ArtGraphMultiTask(Artgraph):
    """
    ArtGraph Dataset for multi task classification.

    Args:
        image_dir: the directory in which images are stored.
        df_image_label: a pandas' dataframe which contains the following column: 'image', 'style', 'genre'. The order is important.
    """

    def __init__(self, image_dir:str, df_image_label: DataFrame, transform_type: str = 'resnet'):
        columns = df_image_label.columns
        assert 'image' in columns and 'style' in columns and 'genre' in columns

        super().__init__(image_dir, df_image_label, transform_type)

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        style_id = self.dataset.iloc[idx, 1]
        genre_id = self.dataset.iloc[idx, 2]

        return image_tensor, [style_id, genre_id]


class ArtGraphSingleTask(Artgraph):
    """
    ArtGraph Dataset for single task classification.

    Args:
        image_dir: the directory in which images are stored.
        df_image_label: a pandas' dataframe which contains the following column: 'image', <label_name>. The order is important.
    """

    def __init__(self, image_dir:str, df_image_label: DataFrame, transform_type: str ='resnet'):
        assert 'image' in df_image_label.columns
        super().__init__(image_dir, df_image_label, transform_type)

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        label_id = self.dataset.iloc[idx, 1]

        return image_tensor, label_id


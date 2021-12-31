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

class MultiModalArtgraph(data.Dataset):
    def __init__(self, image_dir:str, df_image_label: DataFrame, embeddings: Tensor):
        self.image_dir = image_dir
        self.dataset = df_image_label
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.dataset)

    def prepare_image(self, image_path:str) -> Tensor:

        image = Image.open(image_path)
        if(image.mode != 'RGB'):
            image = image.convert('RGB')
        image_tensor = transform(image)

        return image_tensor



class MultiModalArtgraphMultiTask(MultiModalArtgraph):

    def __init__(self, image_dir:str, df_image_label: DataFrame, embeddings: Tensor):
        columns = df_image_label.columns
        assert 'image' in columns and 'style' in columns and 'genre' in columns
        assert len(df_image_label) == embeddings.shape[0]

        super().__init__(image_dir, df_image_label, embeddings)

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        style_id = self.dataset.iloc[idx, 1]
        genre_id = self.dataset.iloc[idx, 2]
        embedding = self.embeddings[idx]

        return image_tensor, embedding, [style_id, genre_id]


class MultiModalArtgraphSingleTask(MultiModalArtgraph):

    def __init__(self, image_dir:str, df_image_label: DataFrame, embeddings: Tensor):
        assert 'image' in df_image_label.columns
        assert len(df_image_label) == embeddings.shape[0]
        super().__init__(image_dir, df_image_label, embeddings)

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        label_id = self.dataset.iloc[idx, 1]
        embedding = self.embeddings[idx]

        return image_tensor, embedding, label_id


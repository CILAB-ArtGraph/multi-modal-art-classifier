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
    """A base class dataset for a multi-modal approach for artwork classification.
    In addition to the label, for each artwork is assigned an embedding vector, which encodes
    its contextual information.

    Args:
        image_dir: the directory in which all the image files are located.
        df_image_label: a dataframe consisting of 'image' column (the artwork image path into the image_dir directory),
            labels column.
        embeddings: a tensor of dimension (n_artworks, embedding_size). The vector embedding[n] is assigned to the image
            in the nth row in the df_image_label dataframe.  
    """
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
    """The dataset for a multi-modal multi-task approach for artwork classification.
    """
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
    """The dataset for a multi-modal single-task approach for artwork classification.
    """
    def __init__(self, image_dir:str, df_image_label: DataFrame, embeddings: Tensor, type = 'train', emb_type = 'artwork'):
        assert 'image' in df_image_label.columns
        #assert len(df_image_label) == embeddings.shape[0]
        self.type = type
        self.emb_type = emb_type
        super().__init__(image_dir, df_image_label, embeddings)

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        label_id = self.dataset.iloc[idx, 1]
        if self.type == 'train':
            if self.emb_type == 'artwork':
                embedding = self.embeddings[idx]
            if self.emb_type != 'artwork':
                embedding = self.embeddings[label_id]
        else:
            embedding = self.embeddings[idx]

        return image_tensor, embedding, label_id

class LabelProjectionDataset(MultiModalArtgraph):

    def __init__(self, image_dir:str, df_image_label: DataFrame, embeddings: Tensor, emb_type):
        super().__init__(image_dir, df_image_label, embeddings)
        self.emb_type = emb_type

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        label_id = self.dataset.iloc[idx, 1]
        if self.emb_type == 'artwork':
            emb = self.embeddings[idx]
        else:
            emb = self.embeddings[label_id]

        return image_tensor, emb

class NewMultiModalArtgraphMultiTask:

    def __init__(self, image_dir:str, df_image_label: DataFrame, embedding_style: Tensor, embedding_genre: Tensor, type: str = 'train', emb_type = 'artwork'):
        columns = df_image_label.columns
        assert 'image' in columns and 'style' in columns and 'genre' in columns

        self.image_dir = image_dir
        self.dataset = df_image_label
        self.embedding_style = embedding_style
        self.embedding_genre = embedding_genre
        self.type = type
        self.emb_type = emb_type

    def __len__(self) -> int:
        return len(self.dataset)

    def prepare_image(self, image_path:str) -> Tensor:

        image = Image.open(image_path)
        if(image.mode != 'RGB'):
            image = image.convert('RGB')
        image_tensor = transform(image)

        return image_tensor

    def __getitem__(self, idx: Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir, self.dataset.iloc[idx, 0])
        image_tensor = self.prepare_image(image_path)

        style_id = self.dataset.iloc[idx, 1]
        genre_id = self.dataset.iloc[idx, 2]
        if self.type == 'train':
            if self.emb_type == 'artwork':
                embedding_style = self.embedding_style[idx]
                embedding_genre = self.embedding_genre[idx]
            else:
                embedding_style = self.embedding_style[style_id]
                embedding_genre = self.embedding_genre[genre_id]
        else:
            embedding_style = self.embedding_style[idx]
            embedding_genre = self.embedding_genre[idx]

        return image_tensor, embedding_style, embedding_genre, [style_id, genre_id]


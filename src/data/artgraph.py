import numpy as np
import pandas as pd
import shutil
import torch
import torch_geometric as pyg
import os
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)

class ArtGraph(InMemoryDataset):
    """
    ArtGraph is a Knowledge Graph which models information about the art domain.
    It was introduced for the first time in the following paper https://arxiv.org/abs/2105.15028

    This dataset has been created for a node-clasification task. For each artwork is associated
    the style and the genre class.

    A 128-dim visual feature vector is assigned to each artwork. For the other nodes you can assign
    a one-hot vector or a constant value.

    Args:
        root (string): the root directory where the dataset should be saved.
        preprocess (string): type feature to assign to the other nodes. Can be 'one-hot', 'constant' or
        None for featureless nodes.
        transform (callable): a function that takes the data object and returns a transformed version.
        pre-transform (callable): as 'transform' but the data will be transformed before being saved to disk. 
        features (boolean): if assign (True) or not (False) visual features to artwork nodes.
        type (string): the purpose the graph is used for, that is 'train', 'test' or 'validation'.
    """

    def __init__(self, root, preprocess='one-hot', transform=None,
                 pre_transform=None, features=True, type='train'):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        self.features = features
        self.type = type

        assert self.preprocess in [None, 'constant', 'one-hot']
        assert type in ['train', 'validation', 'test']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        file_names = [
            'node-feat', 'node-label', 'relations',
            'num-node-dict.csv'
        ]

        return file_names

    @property
    def processed_file_names(self):
        return 'none.pt'
    
    def download(self):
        if not all([os.path.exists(f) for f in self.raw_paths]):
            raise Exception('Missed files')

    def process(self):
        data = pyg.data.HeteroData()
        if self.features == True:
            path = os.path.join(self.raw_dir, 'node-feat', 'artwork', 'node-feat.csv')
            x_artwork = pd.read_csv(path, header=None, dtype=np.float32).values
            data['artwork'].x = torch.from_numpy(x_artwork)

        else:
            path = os.path.join(self.raw_dir, 'num-node-dict.csv')
            num_nodes_df = pd.read_csv(path)
            data['artwork'].num_nodes = num_nodes_df['artwork'].tolist()[0]

        path = os.path.join(self.raw_dir, 'node-label', 'artwork', 'node-label-style.csv')
        style_artwork = pd.read_csv(path, header=None, dtype=np.float32).values.flatten()
        data['artwork'].y_style = torch.from_numpy(style_artwork)

        path = os.path.join(self.raw_dir, 'node-label', 'artwork', 'node-label-genre.csv')
        genre_artwork = pd.read_csv(path, header=None, dtype=np.float32).values.flatten()
        data['artwork'].y_genre = torch.from_numpy(genre_artwork)

        
        path = os.path.join(self.raw_dir, 'num-node-dict.csv')
        num_nodes_df = pd.read_csv(path)
        nodes_type = ['artist', 'gallery', 'style', 'genre',  'tag', 'media', 'field', 'movement']
        if self.preprocess is None:
            for node_type in nodes_type:
                data[node_type].num_nodes = num_nodes_df[node_type].tolist()[0]
        elif self.preprocess == 'constant':
            for node_type in nodes_type:
                data[node_type].x = torch.arange(num_nodes_df[node_type].tolist()[0], dtype = torch.float32).unsqueeze(1)
        elif self.preprocess == 'one-hot':
            for node_type in nodes_type:
                data[node_type].x = torch.eye(num_nodes_df[node_type].tolist()[0])
            
        for edge_type in [('artist', 'field', 'field'), 
                          ('artist', 'movement', 'movement'), 
                          ('artist', 'teacher', 'artist'), 
                          ('artwork', 'media', 'media'), 
                          ('artwork', 'about', 'tag'), 
                          ('artwork', 'genre', 'genre'), 
                          ('artwork', 'style', 'style'), 
                          ('artwork', 'author', 'artist'), 
                          ('artwork', 'locatedin', 'gallery')]:
            f = '___'.join(edge_type)
            path = os.path.join(self.raw_dir, 'relations', f, 'edge.csv')
            edge_index = pd.read_csv(path, header=None, dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            h, r, t = edge_type
            edge_type = (h, r + '_rel', t)
            data[edge_type].edge_index = edge_index
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_classes(self): 
        return {
           'style': self.data['style'].x.shape[0],
           'genre': self.data['genre'].x.shape[0]
        }

    @property
    def num_features(self):
        return self.data['artwork'].x.shape[1]



    

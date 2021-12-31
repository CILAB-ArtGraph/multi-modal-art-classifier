from typing import Dict
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

from data.data import ArtGraphSingleTask, ArtGraphMultiTask
from data.data_kg import MultiModalArtgraphSingleTask, MultiModalArtgraphMultiTask

def prepare_raw_dataset(base_dir: str, type : str):
    artwork = pd.read_csv(os.path.join(base_dir, type, 'mapping/artwork_entidx2name.csv'), names=['idx', 'image'])
    style = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-style.csv'), names=['style'])
    genre = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-genre.csv'), names=['genre'])

    dataset = pd.concat([artwork, style, genre], axis=1)
    return dataset

def load_dataset(base_dir: str, image_dir: str, mode : str, label: str = None, transform_type = 'resnet'):
    assert mode in ['single_task', 'multi_task']

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    if mode == 'single_task':
        dataset_train = ArtGraphSingleTask(image_dir, raw_train[['image', label]], transform_type)
        dataset_valid = ArtGraphSingleTask(image_dir, raw_valid[['image', label]], transform_type)
        dataset_test = ArtGraphSingleTask(image_dir, raw_test[['image', label]], transform_type)
    else:
        dataset_train = ArtGraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], transform_type)
        dataset_valid = ArtGraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']], transform_type)
        dataset_test = ArtGraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']], transform_type)
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_kg(base_dir: str, image_dir: str, mode : str, label: str = None):
    assert mode in ['single_task', 'multi_task']

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    embeddings = torch.load(os.path.join(base_dir, 'train', 'gnn_artwork_embeddings.pt'))

    if mode == 'single_task':
        dataset_train = MultiModalArtgraphSingleTask(image_dir, raw_train[['image', label]], embeddings)
        dataset_valid = ArtGraphSingleTask(image_dir, raw_valid[['image', label]])
        dataset_test = ArtGraphSingleTask(image_dir, raw_test[['image', label]])
    else:
        dataset_train = MultiModalArtgraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], embeddings)
        dataset_valid = ArtGraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']])
        dataset_test = ArtGraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']])
    
    return dataset_train, dataset_valid, dataset_test

def prepare_dataloader(datasets: Dict[str, Dataset], batch_size: int, **kwargs):
    train = DataLoader(datasets['train'], batch_size = batch_size, **kwargs)
    valid = DataLoader(datasets['valid'], batch_size = batch_size, **kwargs)
    test = DataLoader(datasets['test'], batch_size = batch_size, **kwargs)

    data_loaders = {
        'train': train,
        'valid': valid,
        'test': test    
    }

    return data_loaders

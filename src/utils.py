from typing import Dict
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from data.data import ArtGraphSingleTask, ArtGraphMultiTask
from data.data_kg import LabelProjectionDataset, MultiModalArtgraphSingleTask, MultiModalArtgraphMultiTask, LabelProjectionDataset

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

def load_dataset_embeddings(base_dir: str, image_dir: str, mode : str, label: str = None):
    assert mode in ['single_task', 'multi_task']

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    embeddings_train = torch.load(os.path.join(base_dir, 'train', f'gnn_{label}_embeddings.pt'))
    embeddings_validation = torch.load(os.path.join(base_dir, 'validation', f'valid_proj_embeddings_{label}.pt'))
    embeddings_test = torch.load(os.path.join(base_dir, 'test', f'test_proj_embeddings_{label}.pt'))

    if mode == 'single_task':
        dataset_train = MultiModalArtgraphSingleTask(image_dir, raw_train[['image', label]], embeddings_train, type = 'train')
        dataset_valid = MultiModalArtgraphSingleTask(image_dir, raw_valid[['image', label]], embeddings_validation, type = 'validation')
        dataset_test = MultiModalArtgraphSingleTask(image_dir, raw_test[['image', label]], embeddings_test, type = 'test')
    else:
        dataset_train = MultiModalArtgraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], embeddings_train)
        dataset_valid = MultiModalArtgraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']], embeddings_validation)
        dataset_test = MultiModalArtgraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']], embeddings_test)
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_projection(base_dir: str, image_dir: str, mode : str, label: str = None):
    assert mode in ['single_task']

    raw = prepare_raw_dataset(base_dir, type = 'train')

    embeddings = torch.load(os.path.join(base_dir, 'train', f'gnn_{label}_embeddings.pt'))

    dataset = LabelProjectionDataset(image_dir, raw[['image', label]], embeddings)

    train_idx, drop_idx = train_test_split(list(range(len(dataset))), test_size = 0.2, random_state=11)
    dataset_train = Subset(dataset, train_idx)
    dataset_drop = Subset(dataset, drop_idx)

    valid_idx, test_idx = train_test_split(list(range(len(dataset_drop))), test_size = 0.5, random_state=11)
    dataset_valid = Subset(dataset_drop, valid_idx)
    dataset_test = Subset(dataset_drop, test_idx)

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

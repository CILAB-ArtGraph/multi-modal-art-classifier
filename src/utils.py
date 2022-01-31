from typing import Dict
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from data.data import ArtGraphSingleTask, ArtGraphMultiTask
from data.data_kg import (LabelProjectionDataset, MultiModalArtgraphSingleTask, MultiModalArtgraphMultiTask, 
                          LabelProjectionDataset, NewMultiModalArtgraphMultiTask)

def prepare_raw_dataset(base_dir: str, type : str):
    """
    Create the raw dataset.

    Get the raw files and put them into pandas dataframe.
    The dataframe consists of four columns: artwork id, artork name file, artwork style id and artwork genre id.

    Args:
        base_dir: the main directory where the raw files are located.
        type: the subdirectory. Should be one from {train, test, validation}.

    Returns:
        dataset: the pandas dataframe dataset.
    """
    artwork = pd.read_csv(os.path.join(base_dir, type, 'mapping/artwork_entidx2name.csv'), names=['idx', 'image'])
    style = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-style.csv'), names=['style'])
    genre = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-genre.csv'), names=['genre'])

    dataset = pd.concat([artwork, style, genre], axis=1)
    return dataset

def load_dataset(base_dir: str, image_dir: str, mode : str, label: str = None, transform_type = 'resnet'):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for an approach that uses only images.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        mode: single_task or multi_task
        label: genre or style. If multitask put None.
        transform_type: image transformation type. It depends by the model used.

    Returns:
        datasets for train, validation and test.
    """
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

def load_dataset_kg(base_dir: str, image_dir: str, mode : str, label: str = None, embedding: str = None):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used in an contextual approach.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        mode: single_task or multi_task
        label: genre or style. If multitask put None.
        embedding: embedding type. One from {genre, style, artwork}.
    Returns:
        datasets for train, validation and test.
    """
    assert mode in ['single_task', 'multi_task']
    assert embedding in ['artwork', 'genre', 'style']

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    if embedding == 'artwork':
        embeddings = torch.load(os.path.join(base_dir, 'train', f'{embedding}_metapath2vec_embs.pt'))
    else: 
        embeddings = torch.load(os.path.join(base_dir, 'train', f'{embedding}_gnn_embs.pt'))

    if mode == 'single_task':
        assert label in ['genre', 'style']

        dataset_train = MultiModalArtgraphSingleTask(image_dir, raw_train[['image', label]], embeddings)
        dataset_valid = ArtGraphSingleTask(image_dir, raw_valid[['image', label]])
        dataset_test = ArtGraphSingleTask(image_dir, raw_test[['image', label]])
    else:
        dataset_train = MultiModalArtgraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], embeddings)
        dataset_valid = ArtGraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']])
        dataset_test = ArtGraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']])
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_embeddings(base_dir: str, image_dir: str, mode : str, label: str, emb_node: str, emb_type: str):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used in a contextual approach that uses
    embedding also in the forward pass.

    During the training, true embedding are provided. During the validation and test phases projected embedding
    are provided.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        mode: single_task or multi_task
        label: genre or style. If multitask put None.
        emb_node: embedding type. One from {genre, style, artwork}.
        emb_type: embedding technique from which embeddings have been generates. E.g.
            gnn, metapath2vec, node2vec.

    Returns:
        datasets for train, validation and test.
    """
    assert mode in ['single_task', 'multi_task']
    assert emb_node in ['artwork', 'genre', 'style', 'both']
    assert emb_type in ['gnn', 'metapath2vec', 'node2vec']

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    if mode == 'single_task':
        assert label in ['genre', 'style']
        embeddings_train = torch.load(os.path.join(base_dir, 'train', f'{emb_node}_{emb_type}_embs.pt'))
        embeddings_validation = torch.load(os.path.join(base_dir, 'validation', f'{emb_node}_{emb_type}_projs.pt'))
        embeddings_test = torch.load(os.path.join(base_dir, 'test', f'{emb_node}_{emb_type}_projs.pt'))

        dataset_train = MultiModalArtgraphSingleTask(image_dir, raw_train[['image', label]], embeddings_train, type = 'train')
        dataset_valid = MultiModalArtgraphSingleTask(image_dir, raw_valid[['image', label]], embeddings_validation, type = 'validation')
        dataset_test = MultiModalArtgraphSingleTask(image_dir, raw_test[['image', label]], embeddings_test, type = 'test')
    else:
        embeddings_train_genre = torch.load(os.path.join(base_dir, 'train', f'genre_{emb_type}_embs.pt'))
        embeddings_train_style = torch.load(os.path.join(base_dir, 'train', f'style_{emb_type}_embs.pt'))

        embeddings_validation_genre = torch.load(os.path.join(base_dir, 'validation', f'genre_{emb_type}_projs.pt'))
        embeddings_validation_style = torch.load(os.path.join(base_dir, 'validation', f'style_{emb_type}_projs.pt'))

        embeddings_test_genre = torch.load(os.path.join(base_dir, 'test', f'genre_{emb_type}_projs.pt'))
        embeddings_test_style = torch.load(os.path.join(base_dir, 'test', f'style_{emb_type}_projs.pt'))

        dataset_train = NewMultiModalArtgraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], embeddings_train_style, embeddings_train_genre, 'train')
        dataset_valid = NewMultiModalArtgraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']], embeddings_validation_style,embeddings_validation_genre, 'valid')
        dataset_test = NewMultiModalArtgraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']], embeddings_test_style, embeddings_test_genre, 'test')
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_projection(base_dir: str, image_dir: str, node_embedding_name: str, label: str,):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used to train the projector function.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        mode: single_task or multi_task.
        node_embedding_name: node embedding gile name.
        label: the embedding type (e.g. genre, style or artwork).

    Returns:
        datasets for train, validation and test.
    """

    raw = prepare_raw_dataset(base_dir, type = 'train')

    embeddings = torch.load(os.path.join(base_dir, 'train', node_embedding_name))

    dataset = LabelProjectionDataset(image_dir, raw[['image', label]], embeddings)

    train_idx, drop_idx = train_test_split(list(range(len(dataset))), test_size = 0.2, random_state=11)
    dataset_train = Subset(dataset, train_idx)
    dataset_drop = Subset(dataset, drop_idx)

    valid_idx, test_idx = train_test_split(list(range(len(dataset_drop))), test_size = 0.5, random_state=11)
    dataset_valid = Subset(dataset_drop, valid_idx)
    dataset_test = Subset(dataset_drop, test_idx)

    return dataset_train, dataset_valid, dataset_test

def prepare_dataloader(datasets: Dict[str, Dataset], batch_size: int, **kwargs):
    """
    
    """
    train = DataLoader(datasets['train'], batch_size = batch_size, **kwargs)
    valid = DataLoader(datasets['valid'], batch_size = batch_size, **kwargs)
    test = DataLoader(datasets['test'], batch_size = batch_size, **kwargs)

    data_loaders = {
        'train': train,
        'valid': valid,
        'test': test    
    }

    return data_loaders

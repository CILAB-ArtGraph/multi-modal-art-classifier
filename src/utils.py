from typing import Dict
import pandas as pd
import argparse
import os
import mlflow
from functools import wraps
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import config
from data.data import ArtGraphSingleTask, ArtGraphMultiTask
from data.data_kg import (LabelProjectionDataset, MultiModalArtgraphSingleTask, MultiModalArtgraphMultiTask, 
                          LabelProjectionDataset, NewMultiModalArtgraphMultiTask)

def prepare_raw_dataset(base_dir: str, type : str):
    """
    Create the raw dataset.

    Get the raw files and put them into pandas dataframe.
    The dataframe consists of four columns: artwork id, artwork name file, artwork style id and artwork genre id.

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

def load_dataset(base_dir: str, image_dir: str, mode : str, label: str = None, transform_type: str = 'resnet'):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for an approach that uses only images.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        mode: single_task or multi_task
        label: genre or style. If multitask put None.
        transform_type: image transformation type. It depends by the model used. Until now support resnet and vit

    Returns:
        datasets for train, validation and test.
    """
    assert mode in ['single_task', 'multi_task']
    assert transform_type in ['resnet', 'vit']

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

def load_dataset_multimodal(base_dir: str, image_dir: str, mode : str, label: str = None, emb_type: str = None, emb_train: str = None):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used in an contextual approach. This is the dataset used by models like
    ContextNet.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        mode: single_task or multi_task
        label: genre or style, for single task mode.
        emb_type: which node the embeddings encode, may be artwork, style or genre.
        emb_train: the path of the embeddings of training artworks
    Returns:
        datasets for train, validation and test.
    """
    assert mode in ['single_task', 'multi_task']
    assert emb_type in ['artwork', 'genre', 'style']

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    embeddings = torch.load(os.path.join(base_dir, 'train', emb_train))

    if mode == 'single_task':
        assert label in ['genre', 'style']

        dataset_train = MultiModalArtgraphSingleTask(image_dir, raw_train[['image', label]], embeddings, emb_type = emb_type)
        dataset_valid = ArtGraphSingleTask(image_dir, raw_valid[['image', label]])
        dataset_test = ArtGraphSingleTask(image_dir, raw_test[['image', label]])
    else:
        dataset_train = MultiModalArtgraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], embeddings)
        dataset_valid = ArtGraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']])
        dataset_test = ArtGraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']])
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_new_multimodal(base_dir: str, image_dir: str, label: str, emb_type: str, emb_train: str, emb_valid: str, emb_test: str):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used in a contextual approach that uses
    embedding also in the forward pass, in a single-task approach.

    During the training, true embedding are provided. During the validation and test phases projected embedding
    are provided.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        label: genre or style. If multitask put None.
        emb_type: which node the embeddings encode, may be artwork, style or genre.
        emb_train: the path of the (actual) embeddings of train artworks.
        emb_valid: the path of the (projected) embeddings of train artworks.
        emb_test: the path of the (projected) embeddings of train artworks.

    Returns:
        datasets for train, validation and test.
    """

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    embeddings_train = torch.load(os.path.join(base_dir, 'train', emb_train))
    embeddings_validation = torch.load(os.path.join(base_dir, 'validation', emb_valid))
    embeddings_test = torch.load(os.path.join(base_dir, 'test', emb_test))

    dataset_train = MultiModalArtgraphSingleTask(image_dir, raw_train[['image', label]], embeddings_train, type = 'train', emb_type = emb_type)
    dataset_valid = MultiModalArtgraphSingleTask(image_dir, raw_valid[['image', label]], embeddings_validation, type = 'validation', emb_type = emb_type)
    dataset_test = MultiModalArtgraphSingleTask(image_dir, raw_test[['image', label]], embeddings_test, type = 'test', emb_type = emb_type)
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_multitask_new_multimodal(base_dir: str, image_dir: str, emb_type: str, emb_train: Dict[str, str], emb_valid: Dict[str, str], emb_test: Dict[str, str]):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used in a contextual approach that uses
    embedding also in the forward pass, in a multi-task approach.

    During the training, true embedding are provided. During the validation and test phases projected embedding
    are provided.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        emb_type: which node the embeddings encode, may be artwork, style or genre.
        emb_train: the path of the (actual) embeddings of train artworks.
        emb_valid: the path of the (projected) embeddings of train artworks.
        emb_test: the path of the (projected) embeddings of train artworks.

    Returns:
        datasets for train, validation and test.
    """

    raw_train = prepare_raw_dataset(base_dir, type = 'train')
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')

    embeddings_train_genre = torch.load(os.path.join(base_dir, 'train', emb_train['genre']))
    embeddings_train_style = torch.load(os.path.join(base_dir, 'train', emb_train['style']))

    embeddings_validation_genre = torch.load(os.path.join(base_dir, 'validation', emb_valid['genre']))
    embeddings_validation_style = torch.load(os.path.join(base_dir, 'validation', emb_valid['style']))

    embeddings_test_genre = torch.load(os.path.join(base_dir, 'test', emb_test['genre']))
    embeddings_test_style = torch.load(os.path.join(base_dir, 'test', emb_test['style']))

    dataset_train = NewMultiModalArtgraphMultiTask(image_dir, raw_train[['image', 'style', 'genre']], embeddings_train_style, embeddings_train_genre, 'train', emb_type)
    dataset_valid = NewMultiModalArtgraphMultiTask(image_dir, raw_valid[['image', 'style', 'genre']], embeddings_validation_style, embeddings_validation_genre, 'valid', emb_type)
    dataset_test = NewMultiModalArtgraphMultiTask(image_dir, raw_test[['image', 'style', 'genre']], embeddings_test_style, embeddings_test_genre, 'test', emb_type)
    
    return dataset_train, dataset_valid, dataset_test

def load_dataset_projection(base_dir: str, image_dir: str, node_embedding: str, emb_type: str):
    """
    Create a pytorch Dataset (torch.utils.data.Dataset) for being used to train the projector function.

    Args:
        base_dir: the main directory where the raw files are located.
        image_dir: the directory where image files are located.
        node_embedding: node embedding file name.
        label: the embedding type (e.g. genre, style or artwork).
        emb_type: which node the embeddings encode, may be artwork, style or genre.

    Returns:
        datasets for train, validation and test.
    """

    raw = prepare_raw_dataset(base_dir, type = 'train')

    embeddings = torch.load(os.path.join(config.EMBEDDINGS_DIR, node_embedding))

    dataset = LabelProjectionDataset(image_dir, raw[['image', "style", "genre"]], embeddings, emb_type)

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

def tracker(is_tracking, type):
    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            loss, acc, epoch = fun(*args, **kwargs)
            if is_tracking == True:
                mlflow.log_metric(f'{type} loss', loss, step=epoch)
                mlflow.log_metric(f'{type} acc', acc.item(), step=epoch)
            return loss, acc, epoch
        return wrapper
    return decorator

def tracker_multitask(is_tracking, type):
    def decorator(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            loss, acc_style, acc_genre, epoch = fun(*args, **kwargs)
            if is_tracking == True:
                mlflow.log_metric(f'{type} loss', loss, step=epoch)
                mlflow.log_metric(f'{type} acc style', acc_style.item(), step=epoch)
                mlflow.log_metric(f'{type} acc genre', acc_genre.item(), step=epoch)
            return loss, acc_style, acc_genre, epoch
        return wrapper
    return decorator

def track_params(args):
    mlflow.set_experiment(args.exp)
    for arg in vars(args):
        mlflow.log_param(arg, getattr(args, arg))

def get_class_weights(dataset_train, num_classes, label):
    dataset = dataset_train.dataset
    n_artworks = dataset.groupby(label).count().image.sum()
    class_distribution = dataset.groupby(label).count()  
    class_distribution['image'] = class_distribution['image'].map(lambda x: n_artworks/(x*num_classes))
    class_weights = torch.Tensor(class_distribution['image'].tolist())

    return class_weights

def get_base_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Experiment name.')
    parser.add_argument('--dataset_path', type=str, default='../dataset', help='Experiment name.')
    parser.add_argument('--exp', type=str, default='test', help='Experiment name.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch', type=int, default=32, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate.')
    parser.add_argument('--with_weights', action='store_false', help='If using class weights for tackling class imabalnces.')
    parser.add_argument('-t', '--tracking', action='store_false', help='If tracking or not with MLFlow.')

    return parser
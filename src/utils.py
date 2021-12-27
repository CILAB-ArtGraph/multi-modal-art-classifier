from typing import Dict
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def prepare_raw_dataset(base_dir: str, type : str):
    artwork = pd.read_csv(os.path.join(base_dir, type, 'mapping/artwork_entidx2name.csv'), names=['idx', 'image'])
    style = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-style.csv'), names=['style'])
    genre = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-genre.csv'), names=['genre'])

    dataset = pd.concat([artwork, style, genre], axis=1)
    return dataset

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

def get_corrects(predicted, labels):
    return predicted.argmax(dim=1).eq(labels).sum()/predicted.shape[0]
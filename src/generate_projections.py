import os
import torch
import pandas as pd
from os import listdir
from os.path import isfile, join

from models.models_kg import LabelProjector
from data.data import ArtGraphSingleTask
import config

def prepare_raw_dataset(base_dir: str, type : str):
    artwork = pd.read_csv(os.path.join(base_dir, type, 'mapping/artwork_entidx2name.csv'), names=['idx', 'image'])
    style = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-style.csv'), names=['style'])
    genre = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-genre.csv'), names=['genre'])

    dataset = pd.concat([artwork, style, genre], axis=1)
    return dataset

def load_dataset(base_dir: str, image_dir: str):
    raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
    raw_test = prepare_raw_dataset(base_dir, type = 'test')
    dataset_valid = ArtGraphSingleTask(image_dir, raw_valid[['image', 'style', 'genre']])
    dataset_test = ArtGraphSingleTask(image_dir, raw_test[['image', 'style', 'genre']])
    
    return dataset_valid, dataset_test

proj_names = [f for f in listdir(config.PROJECTIONS_DIR) if isfile(join(config.PROJECTIONS_DIR, f))]

for proj_name in proj_names:
    model = LabelProjector(128)
    model.load_state_dict(torch.load(os.path.join(config.PROJECTIONS_DIR, proj_name)))
    model = model.to('cuda', non_blocking=True)

    model.eval()

    dataset_valid, dataset_test = load_dataset(
        base_dir = config.DATASET_DIR, image_dir = config.IMAGE_DIR)

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    loader_valid = DataLoader(dataset_valid, batch_size = 32, num_workers = 6, shuffle = False,
                            drop_last = False, pin_memory = True)

    loader_test = DataLoader(dataset_test, batch_size = 32, num_workers = 6, shuffle = False,
                            drop_last = False, pin_memory = True)

    x_validation = torch.zeros((len(dataset_valid), 128))

    with torch.no_grad():
        model.eval()

        total_loss = total_examples = 0

        print('Generating projections for validation artworks...')
        for idx, images in tqdm(enumerate(loader_valid)):
            image, _ = images
            image = image.to('cuda')
            
            with torch.cuda.amp.autocast():
                out = model(image)
            
            x_validation[idx*32 : (idx+1)*32] = out

    torch.save(x_validation, os.path.join(config.DATASET_DIR, 'validation', 'embeddings', proj_name))

    x_test = torch.zeros((len(dataset_test), 128))

    with torch.no_grad():
        model.eval()

        total_loss = total_examples = 0

        print('Generating projections for test artworks...')
        for idx, images in tqdm(enumerate(loader_test)):
            image, _ = images
            image = image.to('cuda')
            
            with torch.cuda.amp.autocast():
                out = model(image)
            
            x_test[idx*32 : (idx+1)*32] = out

    torch.save(x_test, os.path.join(config.DATASET_DIR, 'test', 'embeddings', proj_name))
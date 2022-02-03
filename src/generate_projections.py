import os
import torch
import pandas as pd
from models.models_kg import LabelProjector
from data.data import ArtGraphSingleTask
import argparse

from os import listdir
from os.path import isfile, join
proj_names = [f for f in listdir('../projections') if isfile(join('../projections', f))]

for proj_name in proj_names:
    model = LabelProjector(128)
    model.load_state_dict(torch.load(f'../projections/{proj_name}'))
    model = model.to('cuda', non_blocking=True)

    model.eval()

    def prepare_raw_dataset(base_dir: str, type : str):
        artwork = pd.read_csv(os.path.join(base_dir, type, 'mapping/artwork_entidx2name.csv'), names=['idx', 'image'])
        style = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-style.csv'), names=['style'])
        genre = pd.read_csv(os.path.join(base_dir, type, 'raw/node-label/artwork/node-label-genre.csv'), names=['genre'])

        dataset = pd.concat([artwork, style, genre], axis=1)
        return dataset

    def load_dataset(base_dir: str, image_dir: str, label: str = None):
        raw_valid = prepare_raw_dataset(base_dir, type = 'validation')
        raw_test = prepare_raw_dataset(base_dir, type = 'test')
        dataset_valid = ArtGraphSingleTask(image_dir, raw_valid[['image', label]])
        dataset_test = ArtGraphSingleTask(image_dir, raw_test[['image', label]])
        
        return dataset_valid, dataset_test

    label = 'genre'
    dataset_valid, dataset_test = load_dataset(
        base_dir = '../dataset/final_full', image_dir ='../../images/imagesf2', label = label)

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

        for idx, images in tqdm(enumerate(loader_valid)):
            image, _ = images
            image = image.to('cuda')
            
            with torch.cuda.amp.autocast():
                out = model(image)
            
            x_validation[idx*32 : (idx+1)*32] = out

    torch.save(x_validation, f'valid_{proj_name}')

    x_test = torch.zeros((len(dataset_test), 128))

    with torch.no_grad():
        model.eval()

        total_loss = total_examples = 0

        for idx, images in tqdm(enumerate(loader_test)):
            image, _ = images
            image = image.to('cuda')
            
            with torch.cuda.amp.autocast():
                out = model(image)
            
            x_test[idx*32 : (idx+1)*32] = out

    torch.save(x_test, f'test_{proj_name}')
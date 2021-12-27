import os
import argparse
from pandas.core.frame import DataFrame
from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from data.data import ArtGraphSingleTask
from models.models import ResnetSingleTask, EarlyStopping
from utils import prepare_raw_dataset, prepare_dataloader

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Experiment name.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Experiment name.')
parser.add_argument('--exp', type=str, default='test', help='Experiment name.')
parser.add_argument('--type', type=str, default='no-kg', help='(no-kg|with-kg)')
parser.add_argument('--mode', type=str, default='single_task', help='Training mode (multi_task|single_task).')
parser.add_argument('--label', type=str, default='genre', help='Label to predict (all|style|genre).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=10**(-3), help='Initial learning rate.')
parser.add_argument('--embsize', type=int, default=128, help='Input graph embedding size')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--activation', type=str, default='relu', help='The activation function.')
args = parser.parse_args()

raw_train = prepare_raw_dataset(args.dataset_path, type = 'train')
raw_valid = prepare_raw_dataset(args.dataset_path, type = 'validation')
raw_test = prepare_raw_dataset(args.dataset_path, type = 'test')

if args.mode == 'single_task':
    dataset_train = ArtGraphSingleTask(args.image_path, raw_train[['image', args.label]])
    dataset_valid = ArtGraphSingleTask(args.image_path, raw_valid[['image', args.label]])
    dataset_test = ArtGraphSingleTask(args.image_path, raw_test[['image', args.label]])
else:
    dataset_train = ArtGraphSingleTask(args.image_path, raw_train[['image', 'style', 'genre']])
    dataset_valid = ArtGraphSingleTask(args.image_path, raw_valid[['image', 'style', 'genre']])
    dataset_test = ArtGraphSingleTask(args.image_path, raw_test[['image', 'style', 'genre']])


data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

num_classes = {
    'genre': 18,
    'style': 32
}

model = ResnetSingleTask(num_classes['genre'])
model = model.to('cuda', non_blocking=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

early_stop = EarlyStopping(patience = 3, min_delta = 0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

def train():
    model.train()

    total_loss = total_correct = total_examples = 0 
    for images, labels in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item() * images.size(0)
            total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
            total_examples = total_examples + len(images)

    return total_loss/total_examples, total_correct/total_examples

@torch.no_grad()
def test(type: str):
    assert type in ['valid', 'test']
    model.eval()

    total_loss = total_correct = total_examples = 0

    for images, labels in tqdm(data_loaders[type]):
        images = images.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)

            if type == 'valid':
                loss = criterion(out, labels)
                total_loss = total_loss + loss.item() * images.size(0)

            total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples
    epoch_acc = total_correct/total_examples

    if type == 'valid':
        early_stop(epoch_acc)
        scheduler.step(epoch_loss)


    return epoch_loss, epoch_acc

epochs = 10
for _ in range(epochs):
    loss, acc = train()
    print(f'Train loss: {loss}; train accuracy: {acc}')
    loss, acc = test('valid')
    print(f'Validation loss: {loss}; validation accuracy: {acc}')

    if early_stop.stop:
        break

_, acc = test('test')
print(f'Test accuracy: {acc}')
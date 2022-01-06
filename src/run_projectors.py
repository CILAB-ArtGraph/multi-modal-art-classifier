import argparse
from torch.nn.modules import dropout
from tqdm import tqdm
import torch
import mlflow

from models.models_kg import LabelProjector
from models.models import EarlyStopping
from utils import load_dataset_projection, prepare_dataloader

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Image folder path.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Dataset path.')
parser.add_argument('--exp', type=str, default='label projections', help='Experiment name.')
parser.add_argument('--label', type=str, default='style', help='Label projection. Options: (style|genre).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='The batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset_projection(
    base_dir = args.dataset_path, image_dir = args.image_path, mode = 'single_task', label = args.label)

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

model = LabelProjector(emb_size = 128)
model = model.to('cuda', non_blocking=True)

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = f'checkpoint_projector.pt')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, mode = 'min', patience = 2, verbose = True)

def train():
    model.train()

    total_loss = total_examples = 0 
    for images, label_embedding in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        label_embedding = label_embedding.to('cuda', non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, label_embedding)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item() * images.size(0)
            total_examples = total_examples + len(images)

    return total_loss/total_examples


@torch.no_grad()
def valid():
    model.eval()

    total_loss = total_examples = 0

    for images, label_embedding in tqdm(data_loaders['valid']):
        images = images.to('cuda', non_blocking=True)
        label_embedding = label_embedding.to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, label_embedding)
            total_loss = total_loss + loss.item() * images.size(0)

            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples

    early_stop(epoch_loss, model)
    scheduler.step(epoch_loss)


    return epoch_loss

@torch.no_grad()
def test():
    model = LabelProjector(emb_size = 128)
    model.load_state_dict(torch.load('checkpoint_projector.pt'))
    model = model.to('cuda', non_blocking=True)

    total_loss = total_examples = 0

    for images, label_embedding in tqdm(data_loaders['test']):
        images = images.to('cuda', non_blocking=True)
        label_embedding = label_embedding.to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, label_embedding)
            total_loss = total_loss + loss.item() * images.size(0)

            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples

    return epoch_loss

mlruns_path = 'file:///home/jbananafish/Desktop/Master/Thesis/code/art-classification-multimodal/tracking/mlruns'
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment(args.exp)
with mlflow.start_run() as run:
    mlflow.log_param('label', args.label)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('batch size', args.batch)
    mlflow.log_param('learning rate', args.lr)

    for epoch in range(args.epochs):
        loss = train()
        print(f'Train loss: {loss}')
        mlflow.log_metric(f'train loss', loss, step=epoch)
        loss = valid()
        print(f'Validation loss: {loss}')
        mlflow.log_metric(f'valid loss', loss, step=epoch)

        if early_stop.stop:
            mlflow.log_param(f'early stop', True)
            break

loss = test()
print(f'Test loss: {loss}')
mlflow.log_metric(f'test loss', loss, step=epoch)
import argparse
from torch.nn.modules import dropout
from tqdm import tqdm
import torch
import mlflow

from models.models_kg import NewMultiModalMultiTask
from models.models import EarlyStopping
from utils import load_dataset_embeddings, prepare_dataloader

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Image folder path.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Dataset path.')
parser.add_argument('--exp', type=str, default='new-multi-modal-multitask', help='Experiment name.')
parser.add_argument('--emb_type', type=str, default='gnn', help='Embedding type (gnn|metapath2vec).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='The batch size.')
parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate.')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset_embeddings(
    base_dir = args.dataset_path, image_dir = args.image_path, mode = 'multi_task', label = 'all', emb_node = 'both', emb_type = args.emb_type)

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

num_classes = {
    'genre': 18,
    'style': 32
}


model = NewMultiModalMultiTask(emb_size = 128, num_classes = num_classes, dropout=0)
model = model.to('cuda', non_blocking=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name = f'{args.emb_type}_new_multi_modal_multitaskcheckpoint.pt'
early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = checkpoint_name)

def train():
    model.train()

    total_loss = total_examples = 0 
    total_style_correct = total_genre_correct = 0
    for images, style_embeddings, genre_embeddings, labels in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        style_labels = labels[0].to('cuda', non_blocking=True)
        genre_labels = labels[1].to('cuda', non_blocking=True)
        style_embeddings = style_embeddings.to('cuda', non_blocking = True)
        genre_embeddings = genre_embeddings.to('cuda', non_blocking = True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(images, style_embeddings, genre_embeddings)

            style_loss = 0.5 * criterion(out[0], style_labels)
            genre_loss = 0.5 * criterion(out[1], genre_labels)
            loss = style_loss + genre_loss
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item() * images.size(0)
        total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
        total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
        total_examples = total_examples + len(images)

    return total_loss/total_examples, total_style_correct/total_examples, total_genre_correct/total_examples


@torch.no_grad()
def valid():
    model.eval()

    total_loss = total_examples = 0
    total_style_correct = total_genre_correct = 0

    for images, style_embeddings, genre_embeddings, labels in tqdm(data_loaders['valid']):
        images = images.to('cuda', non_blocking=True)
        style_labels = labels[0].to('cuda', non_blocking=True)
        genre_labels = labels[1].to('cuda', non_blocking=True)
        style_embeddings = style_embeddings.to('cuda', non_blocking = True)
        genre_embeddings = genre_embeddings.to('cuda', non_blocking = True)

        with torch.cuda.amp.autocast():
            out = model(images, style_embeddings, genre_embeddings)

            style_loss = 0.5 * criterion(out[0], style_labels)
            genre_loss = 0.5 * criterion(out[1], genre_labels)
            loss = style_loss + genre_loss
            total_loss = total_loss + loss.item() * images.size(0)

        total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
        total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
        total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples
    epoch_style_acc = total_style_correct/total_examples
    epoch_genre_acc = total_genre_correct/total_examples

    early_stop(epoch_loss, model)

    return epoch_loss, epoch_style_acc, epoch_genre_acc

@torch.no_grad()
def test():

    model = NewMultiModalMultiTask(emb_size = 128, num_classes = num_classes, dropout=0)
    model.load_state_dict(torch.load(checkpoint_name))
    model = model.to('cuda', non_blocking=True)

    model.eval()

    total_style_correct = total_genre_correct = 0
    total_examples = 0

    for images, style_embeddings, genre_embeddings, labels in tqdm(data_loaders['test']):
        images = images.to('cuda', non_blocking=True)
        style_labels = labels[0].to('cuda', non_blocking=True)
        genre_labels = labels[1].to('cuda', non_blocking=True)
        style_embeddings = style_embeddings.to('cuda', non_blocking = True)
        genre_embeddings = genre_embeddings.to('cuda', non_blocking = True)

        with torch.cuda.amp.autocast():
              out = model(images, style_embeddings, genre_embeddings)

        total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
        total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
        total_examples = total_examples + len(images)

    epoch_style_acc = total_style_correct/total_examples
    epoch_genre_acc = total_genre_correct/total_examples

    return epoch_style_acc, epoch_genre_acc


mlruns_path = 'file:///home/jbananafish/Desktop/Master/Thesis/code/art-classification-multimodal/tracking/mlruns'
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment(args.exp)
with mlflow.start_run() as run:
    mlflow.log_param('emb type', args.emb_type)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('batch size', args.batch)
    mlflow.log_param('learning rate', args.lr)
    mlflow.log_param('net description', "concatenation")

    for epoch in range(args.epochs):
        loss, style_acc, genre_acc = train()
        print(f'Train loss: {loss}; train style accuracy: {style_acc}; train genre accuracy {genre_acc}')
        mlflow.log_metric(f'train loss', loss, step=epoch)
        mlflow.log_metric(f'train style acc', style_acc.item(), step=epoch)
        mlflow.log_metric(f'train genr acc', genre_acc.item(), step=epoch)
        loss, style_acc, genre_acc = valid()
        print(f'Validation loss: {loss}; validation style accuracy: {style_acc}; validation genre accuracy {genre_acc}')
        mlflow.log_metric(f'validation loss', loss, step=epoch)
        mlflow.log_metric(f'validation style acc', style_acc.item(), step=epoch)
        mlflow.log_metric(f'validation genr acc', genre_acc.item(), step=epoch)
        
        if early_stop.stop:
            mlflow.log_param(f'early stop', True)
            break

    style_acc, genre_acc = test()
    print(f'Test style accuracy: {style_acc}; test genre accuracy: {genre_acc}')
    mlflow.log_metric(f'test style acc', style_acc.item())
    mlflow.log_metric(f'test genre acc', genre_acc.item())
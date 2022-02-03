import argparse
from tqdm import tqdm
import torch
import mlflow

from models.models_kg import NewMultiModalSingleTask
from models.models import EarlyStopping
from utils import load_dataset_new_multimodal, prepare_dataloader

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Image folder path.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Dataset path.')
parser.add_argument('--exp', type=str, default='new-multi-modal', help='Experiment name.')
parser.add_argument('--label', type=str, default='genre', help='Label to predict. Options: (style|genre).')
parser.add_argument('--emb_desc', type=str, default='genre', help='(gnn|metapath2vec).')
parser.add_argument('--emb_type', type=str, default='artwork', help='Embedding type (artwork|genre|style).')
parser.add_argument('--emb_train', type=str, default='gnn_artwork_genre_embs_graph.pt', help='Embedding train file name.')
parser.add_argument('--emb_valid', type=str, default='gnn_artwork_genre_valid_embs_graph.pt', help='Embedding train file name.')
parser.add_argument('--emb_test', type=str, default='gnn_artwork_genre_test_embs_graph.pt', help='Embedding train file name.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='The batch size.')
parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset_new_multimodal(
    base_dir = args.dataset_path, image_dir = args.image_path, label = args.label, emb_type = args.emb_type,
    emb_train = args.emb_train, emb_valid = args.emb_valid, emb_test = args.emb_test)

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)
                                                  
num_classes = {
    'genre': 18,
    'style': 32
}

model = NewMultiModalSingleTask(emb_size = 128, num_class = num_classes[args.label], dropout=args.dropout)
model = model.to('cuda', non_blocking=True)

class_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name = f'{args.label}_new_multi_modal_checkpoint.pt'
early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = checkpoint_name)

def train():
    model.train()

    total_loss = total_correct = total_examples = 0 
    for images, embeddings, labels in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        embeddings = embeddings.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(images, embeddings)

            loss = class_criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item() * images.size(0)
            total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
            total_examples = total_examples + len(images)

    return total_loss/total_examples, total_correct/total_examples


@torch.no_grad()
def valid():
    model.eval()

    total_loss = total_correct = total_examples = 0

    for images, embeddings, labels in tqdm(data_loaders['valid']):
        images = images.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)
        embeddings = embeddings.to('cuda', non_blocking = True)

        with torch.cuda.amp.autocast():
            out = model(images, embeddings)

            loss = class_criterion(out, labels)
            total_loss = total_loss + loss.item() * images.size(0)

            total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples
    epoch_acc = total_correct/total_examples

    early_stop(-epoch_acc, model)

    return epoch_loss, epoch_acc

@torch.no_grad()
def test():

    model = NewMultiModalSingleTask(emb_size = 128, num_class = num_classes[args.label], dropout=args.dropout)
    model.load_state_dict(torch.load(checkpoint_name))
    model = model.to('cuda', non_blocking=True)

    model.eval()

    total_correct = total_examples = 0

    for images, embeddings, labels in tqdm(data_loaders['test']):
        images = images.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)
        embeddings = embeddings.to('cuda', non_blocking = True)

        with torch.cuda.amp.autocast():
            out = model(images, embeddings)

            total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
            total_examples = total_examples + len(images)

    epoch_acc = total_correct/total_examples

    return epoch_acc

mlflow.set_experiment(args.exp)
with mlflow.start_run() as run:
    mlflow.log_param('label', args.label)
    mlflow.log_param('emb desc', args.emb_desc)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('batch size', args.batch)
    mlflow.log_param('learning rate', args.lr)
    mlflow.log_param('dropout', args.dropout)
    mlflow.log_param('net description', "concatenation")

    for epoch in range(args.epochs):
        loss, acc = train()
        print(f'Train loss: {loss}; train accuracy: {acc.item()}')
        mlflow.log_metric(f'train loss', loss, step=epoch)
        mlflow.log_metric(f'train acc', acc.item(), step=epoch)
        loss, acc = valid()
        print(f'Validation loss: {loss}; validation accuracy: {acc.item()}')
        mlflow.log_metric(f'valid loss', loss, step=epoch)
        mlflow.log_metric(f'valid acc', acc.item(), step=epoch)

        if early_stop.stop:
            mlflow.log_param(f'early stop', True)
            break

    acc = test()
    print(f'Test accuracy: {acc.item()}')
    mlflow.log_metric(f'test acc', acc.item())
import argparse
from tqdm import tqdm
import torch
import mlflow

from data.data import ArtGraphSingleTask
from models.models import ResnetSingleTask, ResnetMultiTask, EarlyStopping
from utils import prepare_raw_dataset, prepare_dataloader

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Experiment name.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Experiment name.')
parser.add_argument('--exp', type=str, default='resnet-baseline', help='Experiment name.')
parser.add_argument('--mode', type=str, default='single_task', help='Training mode (multi_task|single_task).')
parser.add_argument('--label', type=str, default='genre', help='Label to predict (all|style|genre).')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--freeze', action='store_true', default='False', help='Freeze first three layers.')
#parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
#parser.add_argument('--embsize', type=int, default=128, help='Input graph embedding size')
#parser.add_argument('--activation', type=str, default='relu', help='The activation function.')
args = parser.parse_args()

num_classes = {
    'genre': 18,
    'style': 32
}

raw_train = prepare_raw_dataset(args.dataset_path, type = 'train')
raw_valid = prepare_raw_dataset(args.dataset_path, type = 'validation')
raw_test = prepare_raw_dataset(args.dataset_path, type = 'test')

if args.mode == 'single_task':
    dataset_train = ArtGraphSingleTask(args.image_path, raw_train[['image', args.label]])
    dataset_valid = ArtGraphSingleTask(args.image_path, raw_valid[['image', args.label]])
    dataset_test = ArtGraphSingleTask(args.image_path, raw_test[['image', args.label]])

    model = ResnetSingleTask(num_classes[args.label], freeze=args.freeze)
else:
    dataset_train = ArtGraphSingleTask(args.image_path, raw_train[['image', 'style', 'genre']])
    dataset_valid = ArtGraphSingleTask(args.image_path, raw_valid[['image', 'style', 'genre']])
    dataset_test = ArtGraphSingleTask(args.image_path, raw_test[['image', 'style', 'genre']])

    model = ResnetMultiTask(num_classes, freeze=args.freeze)


data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

model = model.to('cuda', non_blocking=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.epochs)

early_stopping = EarlyStopping(patience = 1, min_delta = 0.001)
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

    if type == 'valid': #update valid loss
        early_stopping(epoch_loss)
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


mlruns_path = 'file:///home/jbananafish/Desktop/Master/Thesis/code/art-classification-multimodal/tracking/mlruns'
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment(args.exp)

with mlflow.start_run() as run:
    mlflow.log_param('mode', args.mode)
    mlflow.log_param('label', args.label)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('batch size', args.batch)
    mlflow.log_param('learning_rate', args.lr)
    mlflow.log_param('free first layers', args.freeze)

for epoch in range(args.epochs):

    loss, acc = train()
    mlflow.log_metric('train loss', loss, step=epoch)
    mlflow.log_metric('train accuracy', acc.detach().item(), step=epoch)
    print(f'Train loss: {loss}; train accuracy: {acc}')

    loss, acc = test('valid')
    mlflow.log_metric('valid loss', loss, step=epoch)
    mlflow.log_metric('valid accuracy', acc.detach().item(), step=epoch)
    print(f'Validation loss: {loss}; validation accuracy: {acc}')

    if early_stopping.stop:
        mlflow.log_param('early stopping', True)
        break

_, acc = test('test')
mlflow.log_metric('test accuracy', acc.detach().item())
print(f'Test accuracy: {acc}')
print()
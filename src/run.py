import argparse
from tqdm import tqdm
import torch
import mlflow

from models.models import ResnetSingleTask, EarlyStopping
from utils import load_dataset, prepare_dataloader

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Experiment name.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Experiment name.')
parser.add_argument('--exp', type=str, default='test', help='Experiment name.')
parser.add_argument('--type', type=str, default='no-kg', help='(no-kg|with-kg)')
parser.add_argument('--label', type=str, default='genre', help='Label to predict (style|genre).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset(
    base_dir = args.dataset_path, image_dir = args.image_path, mode = 'single_task', label = args.label, transform_type = 'resnet')

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

num_classes = {
    'genre': 18,
    'style': 32
}

model = ResnetSingleTask(num_classes[args.label])
model = model.to('cuda', non_blocking=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = f'{args.label}_{args.type}_single-task_model_checkpoint.pt')
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
        early_stop(epoch_loss, model)
        scheduler.step(epoch_loss)


    return epoch_loss, epoch_acc

mlruns_path = 'file:///home/jbananafish/Desktop/Master/Thesis/code/art-classification-multimodal/tracking/mlruns'
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_experiment(args.exp)
with mlflow.start_run() as run:
    mlflow.log_param('type', args.type)
    mlflow.log_param('label', args.label)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('batch size', args.batch)
    mlflow.log_param('learning rate', args.lr)
    mlflow.log_param('dropout', True)

    for epoch in range(args.epochs):
        loss, acc = train()
        print(f'Train loss: {loss}; train accuracy: {acc.item()}')
        mlflow.log_metric(f'train loss', loss, step=epoch)
        mlflow.log_metric(f'train acc', acc.item(), step=epoch)
        loss, acc = test('valid')
        print(f'Validation loss: {loss}; validation accuracy: {acc.item()}')
        mlflow.log_metric(f'valid loss', loss, step=epoch)
        mlflow.log_metric(f'valid acc', acc.item(), step=epoch)

        if early_stop.stop:
            mlflow.log_param(f'early stop', True)
            break

    _, acc = test('test')
    print(f'Test accuracy: {acc.item()}')
    mlflow.log_metric(f'test acc', acc.item(), step=epoch)

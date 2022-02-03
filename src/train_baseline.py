import argparse
from tqdm import tqdm
import torch
import mlflow

from models.models import ResnetSingleTask, EarlyStopping
from utils import load_dataset, prepare_dataloader, tracker, track_params, get_class_weights

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Experiment name.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Experiment name.')
parser.add_argument('--exp', type=str, default='test', help='Experiment name.')
parser.add_argument('--label', type=str, default='genre', help='Label to predict (style|genre).')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-4, help='Initial learning rate.')
parser.add_argument('-t', '--tracking', action='store_false', help='If tracking or not with MLFlow')
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

class_weights = get_class_weights(dataset_train, num_classes[args.label],  args.label)

model = ResnetSingleTask(num_classes[args.label])
model = model.to('cuda', non_blocking=True)

criterion = torch.nn.CrossEntropyLoss(class_weights.to('cuda'))
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name = f'{args.label}_baseline_single-task_model_checkpoint.pt'
early_stop = EarlyStopping(patience = 10, min_delta = 0.001, checkpoint_path = checkpoint_name)

@tracker(args.tracking, 'train')
def train(epoch):

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

    return total_loss/total_examples, total_correct/total_examples, epoch

@tracker(args.tracking, 'valid')
def valid(epoch):

    model.eval()

    total_loss = total_correct = total_examples = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loaders['valid']):
            images = images.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(images)

                loss = criterion(out, labels)
                total_loss = total_loss + loss.item() * images.size(0)

                total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
                total_examples = total_examples + len(images)

        epoch_loss = total_loss/total_examples
        epoch_acc = total_correct/total_examples

    early_stop(epoch_loss, model)

    return epoch_loss, epoch_acc, epoch

def test():

    model = ResnetSingleTask(num_classes[args.label])
    model.load_state_dict(torch.load(checkpoint_name))
    model = model.to('cuda', non_blocking=True)

    model.eval()

    total_correct = total_examples = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loaders['test']):
            images = images.to('cuda', non_blocking=True)
            labels = labels.to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(images)

            total_correct = total_correct + out.argmax(dim=1).eq(labels).sum()
            total_examples = total_examples + len(images)

    epoch_acc = total_correct/total_examples

    return epoch_acc

if args.tracking:
    track_params(args)

for epoch in range(args.epochs):
    loss, acc, _ = train(epoch)
    print(f'Train loss: {loss}; train accuracy: {acc.item()}')
    loss, acc, _ = valid(epoch)
    print(f'Validation loss: {loss}; validation accuracy: {acc.item()}')

acc = test()
print(f'Test accuracy: {acc.item()}')
if args.tracking:
    mlflow.log_metric(f'test acc', acc.item())
import argparse
from tqdm import tqdm
import torch
import mlflow
import numpy as np

from models.models import ResnetMultiTask, EarlyStopping
from utils import load_dataset, prepare_dataloader, tracker_multitask, track_params, get_class_weights

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='../../images/imagesf2', help='Experiment name.')
parser.add_argument('--dataset_path', type=str, default='../dataset', help='Dataset path.')
parser.add_argument('--exp', type=str, default='baseline-multitask', help='Experiment name.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=32, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate.')
parser.add_argument('-t', '--tracking', action='store_false', help='If tracking or not with MLFlow')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset(
    base_dir = args.dataset_path, image_dir = args.image_path, mode = 'multi_task', label = None)


data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

num_classes = {
    'genre': 18,
    'style': 32
}

class_weights_genre = get_class_weights(dataset_train, num_classes['genre'], 'genre').to('cuda')
class_weights_style = get_class_weights(dataset_train, num_classes['style'], 'style').to('cuda')

model = ResnetMultiTask(num_classes)
model = model.to('cuda', non_blocking=True)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
criterion_style = torch.nn.CrossEntropyLoss(class_weights_style)
criterion_genre = torch.nn.CrossEntropyLoss(class_weights_genre)

checkpoint_name = f'baseline_multi-task_model_checkpoint.pt'
early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = checkpoint_name)

w_style = 0.6
w_genre = 0.4

@tracker_multitask(args.tracking, 'train')
def train(epoch):
    model.train()

    total_loss = 0
    total_style_correct = total_genre_correct = 0
    total_examples = 0 

    for images, labels in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        style_labels = labels[0].to('cuda', non_blocking=True)
        genre_labels = labels[1].to('cuda', non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(images)

            style_loss = w_style * criterion_style(out[0], style_labels)
            genre_loss = w_genre * criterion_genre(out[1], genre_labels)
            loss = style_loss + genre_loss
        loss.backward()
        optimizer.step()


        total_loss = total_loss + loss.item() * images.size(0)
        total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
        total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
        total_examples = total_examples + len(images)

    return total_loss/total_examples, total_style_correct/total_examples, total_genre_correct/total_examples, epoch

@tracker_multitask(args.tracking, 'valid')
def valid(epoch):
    model.eval()

    total_loss = 0
    total_style_correct = total_genre_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loaders['valid']):
            images = images.to('cuda', non_blocking=True)
            style_labels = labels[0].to('cuda', non_blocking=True)
            genre_labels = labels[1].to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast():
                out = model(images)

                style_loss = w_style * criterion_style(out[0], style_labels)
                genre_loss = w_genre * criterion_genre(out[1], genre_labels)
                loss = style_loss + genre_loss
                total_loss = total_loss + loss.item() * images.size(0)

            total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
            total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples
    epoch_style_acc = total_style_correct/total_examples
    epoch_genre_acc = total_genre_correct/total_examples

    early_stop(epoch_loss, model)

    return epoch_loss, epoch_style_acc, epoch_genre_acc, epoch

@torch.no_grad()
def test():

    model = ResnetMultiTask(num_classes)
    model.load_state_dict(torch.load(checkpoint_name))
    model = model.to('cuda', non_blocking=True)

    model.eval()

    total_style_correct = total_genre_correct = 0
    total_examples = 0

    for images, labels in tqdm(data_loaders['test']):
        images = images.to('cuda', non_blocking=True)
        style_labels = labels[0].to('cuda', non_blocking=True)
        genre_labels = labels[1].to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)

        total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
        total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
        total_examples = total_examples + len(images)

    epoch_style_acc = total_style_correct/total_examples
    epoch_genre_acc = total_genre_correct/total_examples

    return epoch_style_acc, epoch_genre_acc


if args.tracking:
    track_params(args)

for epoch in range(args.epochs):
    loss, style_acc, genre_acc, _ = train(epoch)
    print(f'Train loss: {loss}; train style accuracy: {style_acc}; train genre accuracy {genre_acc}')
    loss, style_acc, genre_acc, _ = valid(epoch)
    print(f'Validation loss: {loss}; validation style accuracy: {style_acc}; validation genre accuracy {genre_acc}')

style_acc, genre_acc = test()
print(f'Test style accuracy: {style_acc}; test genre accuracy: {genre_acc}')
if args.tracking:
    mlflow.log_metric(f'test style acc', style_acc.item())
    mlflow.log_metric(f'test genre acc', genre_acc.item())
import argparse
from tqdm import tqdm
import torch
import mlflow
import os

from models.models_kg import MultiModalMultiTask, ContextNetlMultiTask
from models.models import EarlyStopping
from utils import load_dataset_multimodal, prepare_dataloader, tracker_multitask, track_params, get_class_weights, get_base_arguments
import config

torch.manual_seed(1)

parser = get_base_arguments()
parser.add_argument('--net', type=str, default='multi-modal', help='The architecture. Options: (context-net|multi-modal)')
parser.add_argument('--emb_train', type=str, default='node2vec_artwork_embs_graph.pt', help='Embedding train file.')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset_multimodal(
    base_dir = args.dataset_path, image_dir = args.image_path, mode = 'multi_task', label = None, emb_type = 'artwork', emb_train = args.emb_train)

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

num_classes = {
    'genre': 18,
    'style': 32
}

nets = {
    'context-net': ContextNetlMultiTask,
    'multi-modal': MultiModalMultiTask
}
assert args.net in nets.keys()

model = nets[args.net](emb_size = 128, num_classes = num_classes)
model = model.to('cuda', non_blocking=True)

if args.with_weights:
    class_weights_genre = get_class_weights(dataset_train, num_classes['genre'], 'genre').to('cuda')
    class_weights_style = get_class_weights(dataset_train, num_classes['style'], 'style').to('cuda')
    criterion_style = torch.nn.CrossEntropyLoss(class_weights_style)
    criterion_genre = torch.nn.CrossEntropyLoss(class_weights_genre)
else:
    criterion_style = torch.nn.CrossEntropyLoss()
    criterion_genre = torch.nn.CrossEntropyLoss()

if args.net == 'context-net':
    encoder_criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr = args.lr, momentum = 0.9)
    lamb = 0.9
else:
    encoder_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    lamb = 0.6

checkpoint_name = os.path.join(config.CHECKPOINTS_DIR, f'{args.net}_multi-task_checkpoint.pt')
early_stop = EarlyStopping(patience = 1, min_delta = 0.001, checkpoint_path = checkpoint_name)

@tracker_multitask(args.tracking, 'train')
def train(epoch):
    model.train()

    total_loss = total_examples = 0 
    total_style_correct = total_genre_correct = 0 
    for images, embeddings, labels in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        embeddings = embeddings.to('cuda', non_blocking=True)
        style_labels = labels[0].to('cuda', non_blocking=True)
        genre_labels = labels[1].to('cuda', non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out, graph_proj = model(images)

            style_loss = 0.5 * criterion_style(out[0], style_labels)
            genre_loss = 0.5 * criterion_genre(out[1], genre_labels)
            encoder_loss = encoder_criterion(graph_proj, embeddings)
            combined_loss = (lamb * (style_loss + genre_loss)) + ((1-lamb) * encoder_loss)
            combined_loss.backward()
            optimizer.step()

            total_loss = total_loss + combined_loss.item() * images.size(0)
            total_style_correct = total_style_correct + out[0].argmax(dim=1).eq(style_labels).sum()
            total_genre_correct = total_genre_correct + out[1].argmax(dim=1).eq(genre_labels).sum()
            total_examples = total_examples + len(images)

    return total_loss/total_examples, total_style_correct/total_examples, total_genre_correct/total_examples, epoch


@tracker_multitask(args.tracking, 'valid')
def valid(epoch):
    model.eval()

    total_loss = total_examples = 0
    total_style_correct = total_genre_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loaders['valid']):
            images = images.to('cuda', non_blocking=True)
            style_labels = labels[0].to('cuda', non_blocking=True)
            genre_labels = labels[1].to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast():
                out, _ = model(images)

                style_loss = 0.5 * class_criterion(out[0], style_labels)
                genre_loss = 0.5 * class_criterion(out[1], genre_labels)
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

    model = nets[args.net](emb_size = 128, num_classes = num_classes)
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
            out, _ = model(images)

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
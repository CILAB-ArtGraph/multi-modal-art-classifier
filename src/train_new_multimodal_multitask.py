import os
from torch.nn.modules import dropout
from tqdm import tqdm
import torch
import mlflow

from models.models_kg import NewMultiModalMultiTask, NewMultiModalMultiTaskViT
from models.models import EarlyStopping
from utils import load_dataset_multitask_new_multimodal, prepare_dataloader, tracker_multitask, track_params, get_class_weights, get_base_arguments
import config

torch.manual_seed(1)

parser = get_base_arguments()
parser.add_argument('--emb_desc', type=str, default='new multimodal multitask', help='Experiment description.')
parser.add_argument('--emb_type', type=str, default='genre', help='Embedding type (artwork|genre|style).')
parser.add_argument('--emb_train_genre', type=str, default='gnn_genre_embs_graph.pt', help='Embedding genre train file name.')
parser.add_argument('--emb_valid_genre', type=str, default='gnn_genre_valid_embs_graph.pt', help='Embedding genre valid file name.')
parser.add_argument('--emb_test_genre', type=str, default='gnn_genre_test_embs_graph.pt', help='Embedding genre test file name.')
parser.add_argument('--emb_train_style', type=str, default='gnn_style_embs_graph.pt', help='Embedding style train file name.')
parser.add_argument('--emb_valid_style', type=str, default='gnn_style_valid_embs_graph.pt', help='Embedding style valid file name.')
parser.add_argument('--emb_test_style', type=str, default='gnn_style_test_embs_graph.pt', help='Embedding style test file name.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout.')
parser.add_argument('--architecture', type=str, default='vit', help='Architecture (resnet|vit).')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset_multitask_new_multimodal(
    base_dir = args.dataset_path, image_dir = args.image_path, emb_type = args.emb_type,
    emb_train = {'style': args.emb_train_style, 'genre': args.emb_train_genre},
    emb_valid = {'style': args.emb_valid_style, 'genre': args.emb_valid_genre},
    emb_test = {'style': args.emb_test_style, 'genre': args.emb_test_genre})

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

num_classes = {
    'genre': 18,
    'style': 32
}

if args.architecture == 'resnet':
    model = NewMultiModalMultiTask(emb_size = 128, num_classes = num_classes, dropout=args.dropout)
else:
    model = NewMultiModalMultiTaskViT(emb_size = 128, num_classes = num_classes, dropout=args.dropout)
model = model.to('cuda', non_blocking=True)

if args.with_weights:
    class_weights_genre = get_class_weights(dataset_train, num_classes['genre'], 'genre').to('cuda')
    class_weights_style = get_class_weights(dataset_train, num_classes['style'], 'style').to('cuda')
    criterion_style = torch.nn.CrossEntropyLoss(class_weights_style)
    criterion_genre = torch.nn.CrossEntropyLoss(class_weights_genre)
else:
    criterion_style = torch.nn.CrossEntropyLoss()
    criterion_genre = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name = os.path.join(config.CHECKPOINTS_DIR, f'new-multimodal_multi-task_checkpoint.pt')
early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = checkpoint_name)

@tracker_multitask(args.tracking, 'train')
def train(epoch):
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

            style_loss = 0.5 * criterion_style(out[0], style_labels)
            genre_loss = 0.5 * criterion_genre(out[1], genre_labels)
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

    total_loss = total_examples = 0
    total_style_correct = total_genre_correct = 0

    with torch.no_grad():
        for images, style_embeddings, genre_embeddings, labels in tqdm(data_loaders['valid']):
            images = images.to('cuda', non_blocking=True)
            style_labels = labels[0].to('cuda', non_blocking=True)
            genre_labels = labels[1].to('cuda', non_blocking=True)
            style_embeddings = style_embeddings.to('cuda', non_blocking = True)
            genre_embeddings = genre_embeddings.to('cuda', non_blocking = True)

            with torch.cuda.amp.autocast():
                out = model(images, style_embeddings, genre_embeddings)

                style_loss = 0.5 * criterion_style(out[0], style_labels)
                genre_loss = 0.5 * criterion_genre(out[1], genre_labels)
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


if args.tracking:
    track_params(args)

for epoch in range(args.epochs):
    loss, style_acc, genre_acc, _ = train(epoch)
    print(f'Train loss: {loss}; train style accuracy: {style_acc}; train genre accuracy {genre_acc}')
    loss, style_acc, genre_acc, _= valid(epoch)
    print(f'Validation loss: {loss}; validation style accuracy: {style_acc}; validation genre accuracy {genre_acc}')

style_acc, genre_acc = test()
print(f'Test style accuracy: {style_acc}; test genre accuracy: {genre_acc}')
if args.tracking:
    mlflow.log_metric(f'test style acc', style_acc.item())
    mlflow.log_metric(f'test genre acc', genre_acc.item())
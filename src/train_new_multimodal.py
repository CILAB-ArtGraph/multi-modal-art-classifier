from tqdm import tqdm
import torch
import mlflow
import os

from models.models_kg import NewMultiModalSingleTask
from models.models import EarlyStopping
from utils import load_dataset_new_multimodal, prepare_dataloader, tracker, track_params, get_class_weights, get_base_arguments
import config

torch.manual_seed(1)

parser = get_base_arguments()
parser.add_argument('--label', type=str, default='genre', help='Label to predict. Options: (style|genre).')
parser.add_argument('--emb_desc', type=str, default='genre', help='(gnn|metapath2vec).')
parser.add_argument('--emb_type', type=str, default='genre', help='Embedding type (artwork|genre|style).')
parser.add_argument('--emb_train', type=str, default='gnn_genre_embs_graph.pt', help='Embedding train file name.')
parser.add_argument('--emb_valid', type=str, default='gnn_genre_valid_embs_graph.pt', help='Embedding train file name.')
parser.add_argument('--emb_test', type=str, default='gnn_genre_test_embs_graph.pt', help='Embedding train file name.')
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

if args.with_weights:
    class_weights = get_class_weights(dataset_train, num_classes[args.label],  args.label)
    class_criterion = torch.nn.CrossEntropyLoss(class_weights.to('cuda'))
else:
    class_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name = os.path.join(config.CHECKPOINTS_DIR, f'{args.label}_new-multimodal_single-task_checkpoint.pt')
early_stop = EarlyStopping(patience = 3, min_delta = 0.001, checkpoint_path = checkpoint_name)

@tracker(args.tracking, 'train')
def train(epoch):
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

    return total_loss/total_examples, total_correct/total_examples, epoch


@tracker(args.tracking, 'valid')
def valid(epoch):
    model.eval()

    total_loss = total_correct = total_examples = 0

    with torch.no_grad():
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

    return epoch_loss, epoch_acc, epoch

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
from tqdm import tqdm
import os
import torch
import mlflow

from models.models import ResnetSingleTask, EarlyStopping, ViTSingleTask
from utils import load_dataset, prepare_dataloader, tracker, track_params, get_class_weights, get_base_arguments
import config

torch.manual_seed(1)

parser = get_base_arguments()
parser.add_argument('--label', type=str, default='genre', help='Label to predict (style|genre).')
parser.add_argument('--architecture', type=str, default='resnet', help='Architecture (vit|resnet).')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout.')
args = parser.parse_args()

print(args)

dataset_train, dataset_valid, dataset_test = load_dataset(
    base_dir = args.dataset_path, image_dir = args.image_path, mode = 'single_task', label = args.label, transform_type = args.architecture)

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)                       

num_classes = {
    'genre': 18,
    'style': 32
}

if args.architecture == 'resnet':
    model = ResnetSingleTask(num_classes[args.label], args.dropout)
else:
    model = ViTSingleTask(num_classes[args.label], args.dropout)
model = model.to('cuda', non_blocking=True)

if args.with_weights:
    class_weights = get_class_weights(dataset_train, num_classes[args.label],  args.label)
    criterion = torch.nn.CrossEntropyLoss(class_weights.to('cuda'))
else:
    criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name = os.path.join(config.CHECKPOINTS_DIR, f'{args.label}_{args.architecture}_baseline_single-task_checkpoint.pt')
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

    if args.architecture == 'resnet':
        model = ResnetSingleTask(num_classes[args.label])
    else:
        model = ViTSingleTask(num_classes[args.label])
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
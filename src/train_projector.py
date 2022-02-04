from tabnanny import check
from torch.nn.modules import dropout
from tqdm import tqdm
import torch
import os
import config

from models.models_kg import LabelProjector
from models.models import EarlyStopping
from utils import load_dataset_projection, prepare_dataloader, get_base_arguments

torch.manual_seed(1)

parser = get_base_arguments()
parser.add_argument('--node_embedding', type=str, default='gnn_artwork_genre_embs_graph.pt', help='Node embedding file name.')
parser.add_argument('--emb_type', type=str, default='artwork', help='The embedding node type (artwork|style|genre).')
args = parser.parse_args()

dataset_train, dataset_valid, dataset_test = load_dataset_projection(
    base_dir = args.dataset_path, image_dir = args.image_path, node_embedding = args.node_embedding, emb_type = args.emb_type)

data_loaders = prepare_dataloader({'train': dataset_train, 'valid': dataset_valid, 'test': dataset_test},
                                                  batch_size = args.batch, num_workers = 6, shuffle = True,
                                                  drop_last = False, pin_memory = True)

model = LabelProjector(emb_size = 128)
model = model.to('cuda', non_blocking=True)

criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

checkpoint_name =  f'{args.exp}_checkpoint_projector.pt'
early_stop = EarlyStopping(patience = 1, min_delta = 0.001, checkpoint_path = os.path.join(config.PROJECTIONS_DIR, checkpoint_name))

def train():
    model.train()

    total_loss = total_examples = 0 
    for images, embedding in tqdm(data_loaders['train']):
        images = images.to('cuda', non_blocking=True)
        embedding = embedding.to('cuda', non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, embedding)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item() * images.size(0)
            total_examples = total_examples + len(images)

    return total_loss/total_examples


@torch.no_grad()
def valid():
    model.eval()

    total_loss = total_examples = 0

    for images, label_embedding in tqdm(data_loaders['valid']):
        images = images.to('cuda', non_blocking=True)
        label_embedding = label_embedding.to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, label_embedding)
            total_loss = total_loss + loss.item() * images.size(0)

            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples

    early_stop(epoch_loss, model)


    return epoch_loss

@torch.no_grad()
def test():
    model = LabelProjector(emb_size = 128)
    model.load_state_dict(torch.load(os.path.join(config.PROJECTIONS_DIR, checkpoint_name)))
    model = model.to('cuda', non_blocking=True)

    total_loss = total_examples = 0

    for images, label_embedding in tqdm(data_loaders['test']):
        images = images.to('cuda', non_blocking=True)
        label_embedding = label_embedding.to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            out = model(images)

            loss = criterion(out, label_embedding)
            total_loss = total_loss + loss.item() * images.size(0)

            total_examples = total_examples + len(images)

    epoch_loss = total_loss/total_examples

    return epoch_loss

for epoch in range(args.epochs):
    loss = train()
    print(f'Train loss: {loss}')
    loss = valid()
    print(f'Validation loss: {loss}')

loss = test()
print(f'Test loss: {loss}')
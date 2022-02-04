import torch
import torch.nn.functional as F
import torch_geometric.nn as operators
import torch_geometric.transforms as T
from tqdm import tqdm
import argparse
import os

from data.artgraph import ArtGraph
from models.models_graph import HeteroSGNN
import config

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=str, default='style', help='Label to predict (style|genre).')
parser.add_argument('--operator', type=str, default='GATConv', help='GCN operator.')
parser.add_argument('--lr', type=int, default=0.01, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=50, help='Epochs.')
args = parser.parse_args()

def get_accuracy(predicted, labels):
    return predicted.argmax(dim=1).eq(labels).sum()/predicted.shape[0]

def get_accuracies(predicted, labels):
    accuracies = [] 
    for i, _ in enumerate(labels):
        accuracies.append(get_accuracy(predicted[i]['artwork'], labels[i]))
    return accuracies

def get_loss(predicted, labels):
    return F.nll_loss(predicted, labels.type(torch.LongTensor))

def get_losses(predicted, labels):
    losses = []
    for i, _ in enumerate(labels):
        losses.append(get_loss(predicted[i]['artwork'], labels[i]))
        
    return losses

def hetero_training():
    model.train()
    optimizer.zero_grad()
    _, out = model(data_train.x_dict, data_train.edge_index_dict)

    train_losses = get_losses(out, y_train)
    train_total_loss = sum(train_losses)

    train_total_loss.backward()
    optimizer.step()

    train_accuracies = get_accuracies(out, y_train)

    return out, train_losses, train_accuracies

@torch.no_grad()
def hetero_test():
    model.eval()
    _, out_validation = model(x = data_validation.x_dict, edge_index = data_validation.edge_index_dict)
    _, out_test = model(data_test.x_dict, data_test.edge_index_dict)
    
    val_losses = get_losses(out_validation, y_validation)
    test_losses = get_losses(out_test, y_test)

    val_accuracies = get_accuracies(out_validation, y_validation)
    test_accuracies = get_accuracies(out_test, y_test)

    return val_losses, val_accuracies, test_losses, test_accuracies

def del_some_nodes(data):
    """Delete some specific nodes and relations, if you don't want to generate the
    embeddings on the full ArtGraph.
    """
    del data['gallery']
    del data['field']
    del data['movement']
    del data['artist', 'movement_rel', 'movement']
    del data['artist', 'field_rel', 'field']
    del data['artwork', 'locatedin_rel', 'gallery']
    del data['artist', 'teacher_rel', 'artist']
    del data['genre']
    del data['artwork', 'genre_rel','genre']

def save_embeddings(model, label):
    print('Saving embeddings...')
    import copy
    new_model = copy.deepcopy(model)

    new_model.eval()
    with torch.no_grad():
        emb, _ = new_model(data_train_full.x_dict, data_train_full.edge_index_dict)
    
    torch.save(emb['artwork'], os.path.join(config.EMBEDDINGS_DIR, f"test_gnn_artwork_{label}_embs.pt"))
    torch.save(emb['artwork'], os.path.join(config.EMBEDDINGS_DIR, f"test_gnn_{label}_embs.pt"))
    print('Saved.')


operator_registry = {
    'SAGEConv': operators.SAGEConv,
    'GraphConv': operators.GraphConv,
    'GATConv': operators.GATConv,
    'GCNConv': operators.GCNConv,
    'GINConv': operators.GINConv
}

activation_registry = {
    'relu': torch.nn.ReLU(),
    'prelu': torch.nn.PReLU()
}


base_data_train_full = ArtGraph(os.path.join(config.DATASET_DIR, 'train'), preprocess='one-hot', features=True, type='train')
base_data_train = ArtGraph(os.path.join(config.DATASET_DIR, 'train_train'), preprocess='one-hot', features=True, type='train')
base_data_validation = ArtGraph(os.path.join(config.DATASET_DIR, 'train_validation'), preprocess='one-hot', features=True, type='validation')
base_data_test = ArtGraph(os.path.join(config.DATASET_DIR, 'train_test'), preprocess='one-hot', features=True, type='test')

data_train_full, data_train, data_validation, data_test = base_data_train_full [0], base_data_train[0], base_data_validation[0], base_data_test[0]

data_train_full = T.ToUndirected()(data_train_full)
data_train = T.ToUndirected()(data_train)
data_validation = T.ToUndirected()(data_validation)
data_test = T.ToUndirected()(data_test)

num_classes = {
    'genre': 18,
    'style': 32
}

label = args.label
model = HeteroSGNN(operator=operator_registry[args.operator], 
                activation=activation_registry['relu'],
                aggr='sum', 
                hidden_channels=128, 
                out_channels=num_classes[label], 
                metadata=data_train.metadata(),
                n_layers=2,
                dropout=0.4,
                bn=True,
                skip=False)

y_train_full = torch.stack([data_train_full['artwork'][f'y_{label}']])
y_train = torch.stack([base_data_train[0]['artwork'][f'y_{label}']])
y_validation = torch.stack([base_data_validation[0]['artwork'][f'y_{label}']])
y_test = torch.stack([base_data_test[0]['artwork'][f'y_{label}']])

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

with torch.no_grad(): 
    out = model(data_train.x_dict, data_train.edge_index_dict)

for epoch in tqdm(range(0, args.epochs)):
    out, train_losses, train_accuracies = hetero_training()
    val_losses, val_accuracies, _, _ = hetero_test()
    if epoch % 5 == 0:
        print(f'{label}_train_loss', round(train_losses[0].detach().item(), 4))
        print(f'{label}_train_accuracy', round(train_accuracies[0].item(), 2) * 100)
        print(f'{label}_val_loss', round(val_losses[0].detach().item(), 4))
        print(f'{label}_val_accuracy', round(val_accuracies[0].item(), 2) * 100)

val_losses, val_accuracies, test_losses, test_accuracies = hetero_test()
print(f'{label}_train_loss', round(train_losses[0].detach().item(), 4))
print(f'{label}_train_accuracy', round(train_accuracies[0].item(), 2) * 100)
print(f'{label}_val_loss', round(val_losses[0].detach().item(), 4))
print(f'{label}_val_accuracy', round(val_accuracies[0].item(), 2) * 100)
print(f'{label}_test_loss', round(test_losses[0].detach().item(), 4))
print(f'{label}_test_accuracy', round(test_accuracies[0].item(), 2) * 100)

save_embeddings(model, label)
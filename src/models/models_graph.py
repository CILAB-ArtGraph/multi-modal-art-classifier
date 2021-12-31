import torch
import torch.nn.functional as F
import torch_geometric.nn as operators

class HeteroGNN(torch.nn.Module):
    def __init__(self, operator=operators.SAGEConv, activation=torch.nn.ReLU, hidden_channels=128, out_channels=300, num_layers=1, dropout=0.5, bn=False, skip=False):
        super(HeteroGNN, self).__init__()
        self.dropout = dropout
        self.bn = bn
        self.skip = skip
        self.activation = activation
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = operator((-1, -1), hidden_channels)
            lin = operators.Linear(-1, hidden_channels)
            bn = torch.nn.BatchNorm1d(hidden_channels)
            self.convs.append(conv)
            self.lins.append(lin)
            self.bns.append(bn)
        self.conv_out = operator((-1, -1), out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if self.skip:
                x = conv(x, edge_index) + self.lins[i](x)
            else:
                x = conv(x, edge_index)
                
            if self.bn:
                x = self.bns[i](x)
            x_emb = self.activation(x)

            if self.training:
                x_emb = F.dropout(x_emb, self.dropout)
        x_out = self.conv_out(x_emb, edge_index)
        return x, F.log_softmax(x_out, dim=1)

class HeteroSGNN(torch.nn.Module):
    def __init__(self, operator, activation, aggr, hidden_channels, out_channels, metadata, n_layers, dropout, bn, skip):
        super(HeteroSGNN, self).__init__()
        self.gnn = HeteroGNN(operator, activation, hidden_channels, out_channels, n_layers, dropout, bn, skip)
        self.gnn = operators.to_hetero(self.gnn, metadata, aggr=aggr)

    def forward(self, x, edge_index):
        emb, out_soft = self.gnn(x, edge_index)
        return emb, [out_soft]

class HeteroMGNN(torch.nn.Module):
    def __init__(self, operator, activation, aggr, hidden_channels, out_channels, metadata, n_layers, dropout, bn, skip):
        super(HeteroMGNN, self).__init__()
        self.gnn_artist = HeteroGNN(operator, activation, hidden_channels, out_channels['artist'], n_layers, dropout, bn, skip)
        self.gnn_artist = operators.to_hetero(self.gnn_artist, metadata, aggr=aggr)

        self.gnn_style = HeteroGNN(operator, activation, hidden_channels, out_channels['style'], n_layers, dropout, bn, skip)
        self.gnn_style = operators.to_hetero(self.gnn_style, metadata, aggr=aggr)

        self.gnn_genre = HeteroGNN(operator, activation, hidden_channels, out_channels['genre'], n_layers, dropout, bn, skip)
        self.gnn_genre = operators.to_hetero(self.gnn_genre, metadata,aggr=aggr)

    def forward(self, x, edge_index):
        return [self.gnn_artist(x, edge_index), self.gnn_style(x, edge_index), self.gnn_genre(x, edge_index)]

class HomoGNN(torch.nn.Module):
    def __init__(self, operator=operators.GCNConv, activation=torch.nn.ReLU(), input_channels=128, hidden_channels=16, out_channels=300, num_layers=1, dropout=0.5, bn=False, skip=False):
        super(HomoGNN, self).__init__()
        self.dropout = dropout
        self.skip = skip
        self.bn = bn
        self.activation = activation
        self.convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        #self.convs.append(operator(input_channels, hidden_channels))
        for _ in range(num_layers):
            conv = operator(-1, hidden_channels)
            lin = operators.Linear(-1, hidden_channels)
            bn = torch.nn.BatchNorm1d(hidden_channels)
            self.convs.append(conv)
            self.lins.append(lin)
            self.bns.append(bn)
        self.conv_out = operator(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            if self.skip:
                x = conv(x, edge_index) + self.lins[i](x)
            else:
                x = conv(x, edge_index)
            x = self.bns[i](x)
            x_emb = self.activation(x)
            if self.training:
              x_emb = F.dropout(x, self.dropout)
        x_emb = self.conv_out(x, edge_index)
        return x, F.log_softmax(x_emb, dim=1)

class HomoSGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(HomoSGNN, self).__init__()
        self.gnn = HomoGNN(**kwargs)

    def forward(self, x, edge_index):
        emb, out_soft = self.gnn(x, edge_index)
        return emb, [out_soft]
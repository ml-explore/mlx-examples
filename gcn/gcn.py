import mlx.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def __call__(self, x, adj):
        x = self.linear(x)
        return adj @ x


class GCN(nn.Module):
    def __init__(self, x_dim, h_dim, out_dim, nb_layers=2, dropout=0.5, bias=True):
        super(GCN, self).__init__()

        layer_sizes = [x_dim] + [h_dim] * nb_layers + [out_dim]
        self.gcn_layers = [
            GCNLayer(in_dim, out_dim, bias)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x, adj):
        for layer in self.gcn_layers[:-1]:
            x = nn.relu(layer(x, adj))
            x = self.dropout(x)

        x = self.gcn_layers[-1](x, adj)
        return x

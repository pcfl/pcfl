from torch import nn

class SimpleNet(nn.Module):
    '''
    hypersimplenet for adult and synthetic experiments
    '''
    def __init__(self, input_dim, hidden_dim, cls_hidden_layer, flatten=True):
        super(SimpleNet, self).__init__()
        now_dim = input_dim
        layers = []
        for _ in range(cls_hidden_layer - 1):
            layers.append(nn.Linear(now_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            now_dim = hidden_dim
        layers.append(nn.Linear(now_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.flatten = flatten

    def forward(self, x):
        output = self.mlp(x).flatten()
        if self.flatten:
            output = output.flatten()
        return output

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1 = nn.Linear(512, 256)
        self.layer_hidden2 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, dim_out)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            ['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.relu(x)
        x = self.layer_out(x)
        return x
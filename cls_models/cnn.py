from torch import nn
import torch.nn.functional as F

class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.dropout = nn.Dropout(0.6)
        self.fc3 = nn.Linear(64, num_classes)
        self.cls = num_classes

        self.weight_keys = [['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class CNNCifar100(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.cls = num_classes

        self.weight_keys = [['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop((F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x

class CNNCelebA(nn.Module):
    def __init__(self, num_classes, drop_out_rate=0, channel=32):
        super(CNNCelebA, self).__init__()
        self.channel = channel

        out = 3
        model_list = []
        for _ in range(4):
            model_list.append(nn.Conv2d(in_channels=out, out_channels=channel, kernel_size=3, padding=1))

            model_list.append(nn.MaxPool2d(2, 2))
            model_list.append(nn.ReLU())
            out = channel
        self.features = nn.Sequential(*model_list)
        self.drop = nn.Dropout(drop_out_rate)
        self.fc = nn.Linear(4*4*channel, num_classes)

        self.weight_keys = [['features.0.weight', 'features.0.bias'],
                            ['features.3.weight', 'features.3.bias'],
                            ['features.6.weight', 'features.6.bias'],
                            ['features.9.weight', 'features.9.bias'],
                            ['fc.weight', 'fc.bias'],
                            ]

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4*4*self.channel)
        x = self.drop(x)
        x = self.fc(x)
        return x
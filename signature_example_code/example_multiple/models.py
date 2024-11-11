import torch
import torch.nn as nn
import layers

def fnn(input_size=602, hidden_sizes=(16, 16, 16), output_size=1, activation=nn.ReLU()):
    layers = []
    in_size = input_size
    for size in hidden_sizes:
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    model = nn.Sequential(*layers)
    return model
class sigformer_stride1(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer_stride1, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=15500, hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_layer = nn.Linear(4, 4, bias=False)
        self.linear_layer.weight.data = torch.tensor([[5,0,0,0], [0,5,0,0], [0,0,5,0], [0,0,0,5]], dtype=torch.float32)
        self.linear_layer.weight.requires_grad = False

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :1*i, :], 3) for i in range(1, 1+int(x.size(1)/1))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.linear_layer(x)
        return x
class sigformer_stride5(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer_stride5, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=3100, hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_layer = nn.Linear(4, 4, bias=False)
        self.linear_layer.weight.data = torch.tensor([[5,0,0,0], [0,5,0,0], [0,0,5,0], [0,0,0,5]], dtype=torch.float32)
        self.linear_layer.weight.requires_grad = False

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :5*i, :], 3) for i in range(1, 1+int(x.size(1)/5))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.linear_layer(x)
        return x
class sigformer_stride10(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer_stride10, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=1550, hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_layer = nn.Linear(4, 4, bias=False)
        self.linear_layer.weight.data = torch.tensor([[5,0,0,0], [0,5,0,0], [0,0,5,0], [0,0,0,5]], dtype=torch.float32)
        self.linear_layer.weight.requires_grad = False

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :10*i, :], 3) for i in range(1, 1+int(x.size(1)/10))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.linear_layer(x)
        return x
class sigformer_stride50(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer_stride50, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=310, hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_layer = nn.Linear(4, 4, bias=False)
        self.linear_layer.weight.data = torch.tensor([[5,0,0,0], [0,5,0,0], [0,0,5,0], [0,0,0,5]], dtype=torch.float32)
        self.linear_layer.weight.requires_grad = False

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :50*i, :], 3) for i in range(1, 1+int(x.size(1)/50))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.linear_layer(x)
        return x





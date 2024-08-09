import torch
import torch.nn as nn
try:
    import layers
except:
    import signature_example_code.example_single.layers as layers

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
class cnn(nn.Module):
    def __init__(self):
        '''
        This CNN architecture is from https://doi.org/10.1080/14697688.2019.1654126
        Arg:
            - Input shape: (batch, channel, seq)
        Examples:
            input = torch.rand((64, 1, 500))
            model = cnn()
            output = model(input)
        '''
        super(cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        # (time_grid, in_features): (100,384), (200,896), (300,1408), (400,1792), (500,2304)
        self.dense1 = nn.Linear(in_features=384, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=1)
        self.LRelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.LRelu2 = nn.LeakyReLU(negative_slope=0.3)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout1d(p=0.25)
        self.dropout2 = nn.Dropout1d(p=0.4)
        self.dropout3 = nn.Dropout1d(p=0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.LRelu1(x)
        x = self.pooling(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.LRelu2(x)
        x = self.pooling(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.LRelu1(x)
        x = self.pooling(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.LRelu1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
class deepsignet(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905)
        self.dense = fnn(input_size=3905, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
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
        # truncation order 1,2,3,4,5
        x = layers.Sig(x, 5)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class sigformer(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905); [5, 25, 125, 625, 3125]
        self.attention = layers.MultiHeadAttention(d_k=[5,25,125,625,3125], in_features=[5,25,125,625,3125], heads=5, out_features=3905)
        # set time_grid=100, then (truncation order, input_size): (1,50),(2,300),(3,1550),(4,7800),(5,39050)
        self.dense = fnn(input_size=39050, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

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
        # truncation order 1,2,3,4,5
        x = torch.stack([layers.Sig(x[:, :10*i, :], 5) for i in range(1, 1+int(x.size(1)/10))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class sigformer_s(nn.Module):
    def __init__(self, augment_include_time=True, T=1):
        super(sigformer_s, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        # (truncation order, in_features): (1,2),(2,6),(3,14),(4,30),(5,62)
        self.attention = layers.SingleAttention(d_k=155, in_features=62)
        self.dense = nn.Linear(in_features=155, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        # truncation order 1,2,3,4,5
        x = layers.Sig(x, 5)
        x = self.attention(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

    
class conv3d_sig(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, signature_truncation=2, n_conv3d_kernels=12, n_predictor_vars=4,p_dropout3d=0.1):
        super(conv3d_sig, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=n_predictor_vars, out_channels=n_conv3d_kernels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dropout3d = nn.Dropout3d(p=p_dropout3d)
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905)
        n_features = int(n_conv3d_kernels + augment_include_original * n_predictor_vars + augment_include_time * 1)
        siglen = int((n_features ** (signature_truncation+1) - n_features)/(n_features-1))
        self.dense = fnn(input_size=siglen*20*20, hidden_sizes=(32,32,32,), output_size=1, activation=nn.ReLU())
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.signature_truncation = signature_truncation
    def forward(self, x):
        x = torch.movedim(x, 1, -1)
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, 1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.conv3d(x)
        x = self.dropout3d(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.movedim(x, 1, -1)
        # truncation order 1,2,3,4,5
        x = layers.Sig(x, self.signature_truncation)
        x = torch.flatten(x,start_dim=1)
        x = self.dense(x)
        return x 
    
class conv3d(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, signature_truncation=2, n_conv3d_kernels=12, n_predictor_vars=4,p_dropout3d=0.1):
        super(conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=n_predictor_vars, out_channels=n_conv3d_kernels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dropout3d = nn.Dropout3d(p=p_dropout3d)
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905)
        n_features = int(n_conv3d_kernels + augment_include_original * n_predictor_vars + augment_include_time * 1)
        siglen = int((n_features ** (signature_truncation+1) - n_features)/(n_features-1))
        self.dense = fnn(input_size=n_conv3d_kernels*8*20*20, hidden_sizes=(32,32,32,), output_size=1, activation=nn.ReLU())
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.signature_truncation = signature_truncation
    def forward(self, x):
        x = torch.movedim(x, 1, -1)
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, 1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.conv3d(x)
        x = self.dropout3d(x)
        '''
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        '''
        x = torch.movedim(x, 1, -1)
        # truncation order 1,2,3,4,5
        #x = layers.Sig(x, self.signature_truncation)
        x = torch.flatten(x,start_dim=1)
        x = self.dense(x)
        return x 
        
    
'''
max_output_size = 10
output_cat = None

start=time.time()
for i in range(2):
        hx= rnn(input[:,i,:], (hx_0))
        output.append(hx)

        if output_cat is None:
            output_cat_size = list(hx.size())
            output_cat_size.insert(1, max_output_size)
            output_cat = torch.empty(*output_cat_size, dtype=hx.dtype, device=hx.device)

        output_cat[:, i] = hx
        hx_0 = operation(output_cat[:, 0:i])

print(output_cat)'''    
    
class conv2d_sig(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, signature_truncation=2, n_conv2d_kernels=12, n_predictor_vars=4,p_dropout3d=0.1):
        super(conv2d_sig, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=n_predictor_vars, out_channels=n_conv2d_kernels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        
        self.dropout3d = nn.Dropout3d(p=p_dropout3d)
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905)
        n_features = int(n_conv2d_kernels + augment_include_original * n_predictor_vars + augment_include_time * 1)
        siglen = int((n_features ** (signature_truncation+1) - n_features)/(n_features-1))
        self.dense = fnn(input_size=siglen*20*20, hidden_sizes=(32,32,32,), output_size=1, activation=nn.ReLU())
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.n_conv2d_kernels = n_conv2d_kernels
        self.signature_truncation = signature_truncation
    def forward(self, x):
        x = torch.movedim(x, 1, -1) #move time dim to the end
      
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, 1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-3], x.shape[-2], x.shape[-1])
            
        initshape = (x.shape[0],self.n_conv2d_kernels,x.shape[2],x.shape[3],x.shape[4]) #replace input channel length by output channel length
        conv2d_out = torch.empty(initshape)
        for i in range(x.shape[-1]): #timeaxis
            conv2d_out[:,:,:,:,i] = self.conv2d(x[:,:,:,:,i]) #conv2d on each timestep and concatenate
            
        x = conv2d_out 
        x = self.dropout3d(x)
        
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.movedim(x, 1, -1)
        # truncation order 1,2,3,4,5
        x = layers.Sig(x, self.signature_truncation)
        x = torch.flatten(x,start_dim=1)
        x = self.dense(x)
        return x 
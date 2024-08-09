import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import iisignature
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import iisignature
from itertools import accumulate

#define layers ----->
class SingleAttention(nn.Module):
    def __init__(self, d_k=128, in_features=155):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.in_features = in_features
        self.query = nn.Linear(in_features=self.in_features, out_features=self.d_k, bias=False)
        self.key = nn.Linear(in_features=self.in_features, out_features=self.d_k, bias=False)
        self.value = nn.Linear(in_features=self.in_features, out_features=self.d_k, bias=False)
    def forward(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = torch.div(attn_weights, np.sqrt(self.d_k))
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k=[128, 128, 128], in_features=[5, 25, 125], heads=3, out_features=1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.in_features = in_features
        self.attention_heads = nn.ModuleList([SingleAttention(d_k[i], in_features[i]) for i in range(heads)])
        self.linear = nn.Linear(in_features=sum(d_k), out_features=out_features)
    def forward(self, inputs):
        indice = list(accumulate(self.in_features))
        attn = [self.attention_heads[0](inputs[:, :, 0:indice[0]])]
        for i in range(1, self.heads):
            attn.append(self.attention_heads[i](inputs[:, :, indice[i-1]:indice[i]]))
        concat_attn = torch.cat(attn, dim=-1)
        output = self.linear(concat_attn)
        return output
class SigFn(Function):
    @staticmethod
    def forward(ctx, X, m):
        result = iisignature.sig(X.detach().numpy(), m)
        ctx.save_for_backward(X)
        ctx.m = m
        return torch.FloatTensor(result)
    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        m = ctx.m
        result = iisignature.sigbackprop(grad_output.numpy(), X.detach().numpy(), m)
        return torch.FloatTensor(result), None
class LogSigFn(Function):
    @staticmethod
    def forward(ctx, X, s, method):
        result = iisignature.logsig(X.detach().numpy(), s, method)
        ctx.save_for_backward(X)
        ctx.s = s
        ctx.method = method
        return torch.FloatTensor(result)
    @staticmethod
    def backward(ctx, grad_output):
        (X,) = ctx.saved_tensors
        s = ctx.s
        method = ctx.method
        g = grad_output.numpy()
        result = iisignature.logsigbackprop(g, X.detach().numpy(), s, method)
        return torch.FloatTensor(result), None, None
def Sig(X, m):
    '''
    This is a PyTorch function which just dose iisignature.sig
    Source: https://github.com/bottler/iisignature/tree/master
    '''
    return SigFn.apply(X, m)
def LogSig(X, s, method=""):
    '''
    This is a PyTorch function which just dose iisignature.logsig
    Source: https://github.com/bottler/iisignature/tree/master
    '''
    return LogSigFn.apply(X, s, method)


#define models ----->
class conv3d(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, signature_truncation=2, n_conv3d_kernels=12, n_predictor_vars=4,p_dropout3d=0.1):
        super(conv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=n_predictor_vars, out_channels=n_conv3d_kernels, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dropout3d = nn.Dropout3d(p=p_dropout3d)
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
   
        x = torch.movedim(x, 1, -1)
        x = torch.flatten(x,start_dim=1)
        x = self.dense(x)
        return x 
            
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
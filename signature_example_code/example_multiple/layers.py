import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import iisignature
from itertools import accumulate

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



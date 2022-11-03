import torch
from torch import nn
from torch.nn import init
import math

def attention(Q, K, V, masked=False):
    m = Q.shape[-1]
    A = Q @ K.mT / m
    if masked:
        indices = torch.triu_indices(A.shape[-2], A.shape[-1], offset=1)
        A[..., indices[0], indices[1]] = -torch.inf
    return torch.softmax(A, dim=-1) @ V

def multiHeadAttention(Q, K, V, W_q, W_k, W_v, W_o, masked=False):
    Q = Q.unsqueeze(-3)
    K = K.unsqueeze(-3)
    V = V.unsqueeze(-3)
    S = attention(Q @ W_q, K @ W_k, V @ W_v, masked=masked)
    return torch.sum(S @ W_o, 1)

def layerNorm(x):
    size = x.shape
    mean = x.mean(axis=-1).flatten().repeat_interleave(x.shape[-1]).reshape(size)
    std = x.std(axis=-1).flatten().repeat_interleave(x.shape[-1]).reshape(size)
    return x - mean / (std + 1e-5)

def positionalEncoding(X):
    n_tokens = X.shape[-2]
    d_model = X.shape[-1]
    X = X.reshape(-1, n_tokens, d_model)
    pos = torch.arange(n_tokens).repeat(d_model, 1).T
    progression = torch.arange(0, d_model, 2).repeat_interleave(2).repeat(n_tokens, 1)
    wavelengths = pos / 10000**(progression/d_model)
    wavelengths[:, ::2] = torch.pi/2 - wavelengths[:, ::2]
    return X + torch.cos(wavelengths)
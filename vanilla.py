import torch
from torch import nn
from torch.nn import init
import math
from collections import OrderedDict
from functions import multiHeadAttention, layerNorm, positionalEncoding

DEFAULT_D_MODEL = 512
DEFAULT_N_HEADS = 8
DEFAULT_N_ENCODER_BLOCKS = 6
DEFAULT_N_DECODER_BLOCKS = 6
DEFAULT_P = 0.1

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads=DEFAULT_N_HEADS, d_model=DEFAULT_D_MODEL, masked=False):
        super().__init__()
        self.masked = masked
        m = d_model//n_heads
        self.W_q = nn.Parameter(torch.empty((n_heads, d_model, m)))
        self.W_k = nn.Parameter(torch.empty((n_heads, d_model, m)))
        self.W_v = nn.Parameter(torch.empty((n_heads, d_model, m)))
        self.W_o = nn.Parameter(torch.empty((n_heads, m, d_model)))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_q, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_k, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_v, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_o, a=math.sqrt(5))

    def forward(self, input):
        Q = input
        K = input
        V = input
        return multiHeadAttention(Q, K, V, self.W_q, self.W_k, self.W_v, self.W_o, masked=self.masked)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads=DEFAULT_N_HEADS, d_model=DEFAULT_D_MODEL, masked=False):
        super().__init__()
        self.masked = masked
        m = d_model//n_heads
        self.W_q = nn.Parameter(torch.empty((n_heads, d_model, m)))
        self.W_k = nn.Parameter(torch.empty((n_heads, d_model, m)))
        self.W_v = nn.Parameter(torch.empty((n_heads, d_model, m)))
        self.W_o = nn.Parameter(torch.empty((n_heads, m, d_model)))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W_q, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_k, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_v, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_o, a=math.sqrt(5))

    def forward(self, src, target):
        Q = target
        K = src
        V = src
        return multiHeadAttention(Q, K, V, self.W_q, self.W_k, self.W_v, self.W_o, masked=self.masked)

class FeedForward(nn.Module):
    def __init__(self, d_model=DEFAULT_D_MODEL, activation_function=nn.ReLU()):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty((d_model, d_model)))
        self.W2 = nn.Parameter(torch.empty((d_model, d_model)))
        self.b1 = nn.Parameter(torch.empty(d_model))
        self.b2 = nn.Parameter(torch.empty(d_model))
        self.activation_function = activation_function
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W1)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b1, -bound, bound)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W2)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b2, -bound, bound)

    def forward(self, input):
        return self.activation_function(input @ self.W1 + self.b1) @ self.W2 + self.b2

class EncoderBlock(nn.Module):
    def __init__(self, n_heads=DEFAULT_N_HEADS, d_model=DEFAULT_D_MODEL, p=DEFAULT_P):
        super().__init__()
        self.attention = MultiHeadSelfAttention(n_heads=n_heads, d_model=d_model)
        self.ffn = FeedForward(d_model=d_model)
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, input):
        H = layerNorm(self.dropout1(self.attention(input)) + input)
        return layerNorm(self.dropout2(self.ffn(H)) + H)

class Encoder(nn.Module):
    def __init__(self, n_heads=DEFAULT_N_HEADS, d_model=DEFAULT_D_MODEL,
                 n_encoder_blocks=DEFAULT_N_ENCODER_BLOCKS, p=DEFAULT_P):
        super().__init__()
        blocks = OrderedDict([(f'block{i}', EncoderBlock(n_heads=n_heads, d_model=d_model, p=p))
                     for i in range(n_encoder_blocks)])
        self.blocks = nn.Sequential(blocks)

    def forward(self, input):
        return self.blocks(input)

class DecoderBlock(nn.Module):
    def __init__(self, n_heads=DEFAULT_N_HEADS, d_model=DEFAULT_D_MODEL, p=DEFAULT_P):
        super().__init__()
        self.d_model = d_model
        self.selfAttention = MultiHeadSelfAttention(n_heads=n_heads, d_model=d_model, masked=True)
        self.crossAttention = MultiHeadCrossAttention(n_heads=n_heads, d_model=d_model)
        self.ffn = FeedForward(d_model=d_model)
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)
        self.dropout3 = nn.Dropout(p=p)

    def forward(self, src, target):
        X = src
        Y = target
        H = layerNorm(self.dropout1(self.selfAttention(Y)) + Y)
        Q = layerNorm(self.dropout2(self.crossAttention(X, H)) + H)
        return layerNorm(self.dropout3(self.ffn(Q)) + Q)

class Decoder(nn.Module):
    def __init__(self, n_heads=DEFAULT_N_HEADS, d_model=DEFAULT_D_MODEL,
                 n_decoder_blocks=DEFAULT_N_DECODER_BLOCKS, p=DEFAULT_P):
        super().__init__()
        blocks = [DecoderBlock(n_heads=n_heads, d_model=d_model, p=p)
                     for i in range(n_decoder_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, src, target):
        X = src
        Y = target
        for block in self.blocks:
            Y = block(X, Y)
        return Y

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model=DEFAULT_D_MODEL):
        super().__init__()
        self.coordinates = nn.Parameter(torch.empty((vocab_size, d_model)))
        self.normalization = math.sqrt(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.coordinates, a=math.sqrt(5))

    def forward(self, input):
        return self.normalization * self.coordinates[input]

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, n_encoder_blocks=DEFAULT_N_ENCODER_BLOCKS,
                 n_decoder_blocks=DEFAULT_N_DECODER_BLOCKS, n_heads=DEFAULT_N_HEADS,
                 d_model=DEFAULT_D_MODEL, p=DEFAULT_P):
        super().__init__()
        self.encoder = Encoder(n_encoder_blocks=n_encoder_blocks,
                               n_heads=n_heads, d_model=d_model, p=p)
        self.decoder = Decoder(n_decoder_blocks=n_decoder_blocks,
                               n_heads=n_heads, d_model=d_model, p=p)
        self.inputEmbedding = Embedding(input_vocab_size, d_model=d_model)
        self.outputEmbedding = Embedding(output_vocab_size, d_model=d_model)
        self.dropout1 = nn.Dropout(p=p)
        self.dropout2 = nn.Dropout(p=p)

    def forward(self, src, target):
        X = self.dropout1(positionalEncoding(self.inputEmbedding(src)))
        Y = self.dropout2(positionalEncoding(self.outputEmbedding(target)))
        H = self.encoder(X)
        D = self.decoder(H, Y)
        output = D @ self.outputEmbedding.coordinates.T
        return output
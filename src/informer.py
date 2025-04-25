# informer.py （替换原文档3）
import torch
import torch.nn as nn
import math
from torch import Tensor
import torch.nn.functional as F

class SimplifiedInformer(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, 
                 n_heads=8, e_layers=3, d_layers=2, d_ff=256, 
                 dropout=0.1, use_relative_pos=True):
        super(SimplifiedInformer, self).__init__()
        
        # Enhanced Embedding Layers
        self.enc_embedding = EnhancedEmbedding(input_size, hidden_size, dropout)
        self.dec_embedding = EnhancedEmbedding(input_size, hidden_size, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, dropout, use_relative_pos), 
                        hidden_size, n_heads
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, dropout, use_relative_pos), 
                        hidden_size, n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, dropout, use_relative_pos), 
                        hidden_size, n_heads
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size)
        )
        
        # Enhanced Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, output_size)
        )
        
        # Learnable Exponential Layer
        self.exp_layer = LearnableExponentialLayer()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)

    def forward(self, x):
        # Encoder
        enc_out = self.enc_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        
        # Decoder
        dec_out = self.dec_embedding(x)
        dec_out = self.decoder(dec_out, enc_out)
        dec_out = self.projection(dec_out)
        
        return self.exp_layer(dec_out)

class EnhancedEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(EnhancedEmbedding, self).__init__()
        self.value_conv = nn.Sequential(
            nn.Conv1d(c_in, d_model//2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.value_linear = nn.Linear(d_model//2, d_model)
        self.position = LearnablePositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Feature extraction
        x = x.permute(0, 2, 1)  # [batch, feat, seq]
        x = self.value_conv(x)
        x = x.permute(0, 2, 1)  # [batch, seq, feat]
        x = self.value_linear(x)
        
        # Position encoding
        x = x + self.position(x)
        return self.dropout(self.norm(x))

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return self.pos_embed(positions).unsqueeze(0).expand_as(x)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, dropout=0.1, use_relative_pos=False):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout)
        self.use_relative_pos = use_relative_pos
        
        if use_relative_pos:
            self.relative_bias = nn.Parameter(torch.randn(1, 8, 1, 100)*0.02)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        scale = 1./math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Add relative position bias
        if self.use_relative_pos and L == S:
            scores += self.relative_bias[:, :, :, :L]
        
        if self.mask_flag:
            if L != S:
                mask = torch.ones(B, 1, L, S, device=queries.device).tril()
                scores.masked_fill_(mask == 0, -1e9)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()

class LearnableExponentialLayer(nn.Module):
    def __init__(self):
        super(LearnableExponentialLayer, self).__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.exp(x * self.scale + self.bias)

# ========== Maintain Original Structure ==========
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        d_k = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        new_x = self.dropout(F.gelu(self.conv1(x.transpose(-1,1))))
        new_x = self.dropout(self.conv2(new_x).transpose(-1,1))
        x = x + new_x
        x = self.norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x, None

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross))
        x = self.norm2(x)
        new_x = self.dropout(F.gelu(self.conv1(x.transpose(-1,1))))
        new_x = self.dropout(self.conv2(new_x).transpose(-1,1))
        x = x + new_x
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross):
        for layer in self.layers:
            x = layer(x, cross)
        if self.norm is not None:
            x = self.norm(x)
        return x
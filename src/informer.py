import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Enhanced Attention with Sparse or Multi-Scale Attention
class SparseAttention(nn.Module):
    def __init__(self, mask_flag=False, dropout=0.1, use_relative_pos=False):
        super(SparseAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout)
        self.use_relative_pos = use_relative_pos

        if use_relative_pos:
            self.relative_bias = nn.Parameter(torch.randn(1, 8, 1, 100) * 0.02)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        scale = 1. / math.sqrt(E)
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

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)  # 获取输入序列的长度
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return self.pos_embed(positions).unsqueeze(0).expand(x.size(0), seq_len, -1)  # 扩展到 [batch_size, seq_len, hidden_size]

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape  # B: batch_size, L: seq_len
        _, S, _ = keys.shape  # S: seq_len for keys/values
        H = self.n_heads  # Number of attention heads

        # Project queries, keys, and values to the correct number of heads and dimensions
        queries = self.query_projection(queries).view(B, L, H, -1)  # [B, L, H, depth]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [B, S, H, depth]
        values = self.value_projection(values).view(B, S, H, -1)  # [B, S, H, depth]

        # Perform the attention operation
        out = self.inner_attention(queries, keys, values)
        
        # Reshape the output to [B, L, hidden_size] after attention
        out = out.view(B, L, -1)
        
        # Project back to the original hidden size
        out = self.out_projection(out)
        
        return out

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
        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1):
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
        new_x = self.dropout(F.gelu(self.conv1(x.transpose(-1, 1))))
        new_x = self.dropout(self.conv2(new_x).transpose(-1, 1))
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

class EnhancedInformer(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1,
                 n_heads=16, e_layers=4, d_layers=3, d_ff=512,
                 dropout=0.1, use_relative_pos=True):
        super(EnhancedInformer, self).__init__()

        # Encoder with Sparse Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        SparseAttention(False, dropout, use_relative_pos),  # Sparse Attention
                        hidden_size, n_heads
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size)
        )

        # Decoder with Multi-Scale Attention
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        SparseAttention(True, dropout, use_relative_pos),
                        hidden_size, n_heads
                    ),
                    AttentionLayer(
                        SparseAttention(False, dropout, use_relative_pos),
                        hidden_size, n_heads
                    ),
                    hidden_size,
                    d_ff,
                    dropout=dropout
                ) for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(hidden_size)
        )

        # Learnable Position Embedding
        self.pos_embedding = LearnablePositionalEncoding(hidden_size)

        # Final Output Layer
        self.projection = nn.Linear(hidden_size, output_size)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name:
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0.0)

    def forward(self, x):
        # Add Position Embedding
        x = x + self.pos_embedding(x)

        # Encoder
        enc_out = self.encoder(x)

        # Decoder
        dec_out = self.decoder(enc_out, enc_out)

        # Final Output
        return self.projection(dec_out)

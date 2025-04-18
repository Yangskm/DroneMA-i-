# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]

# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1):
#         super().__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.dropout = nn.Dropout(attention_dropout)
        
#     def _prob_QK(self, Q, K, sample_k, n_top):
#         B, H, L_Q, E = Q.shape
#         _, _, L_K, _ = K.shape
        
#         # 确保n_top不超过序列长度
#         actual_n_top = min(n_top, L_Q)
        
#         K_expand = K.unsqueeze(2).expand(-1, -1, L_Q, -1, -1)
        
#         # 确保sample_k不超过L_K
#         actual_sample_k = min(sample_k, L_K)
#         index_sample = torch.randint(L_K, (L_Q, actual_sample_k)).to(Q.device)
        
#         K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1).to(Q.device), index_sample, :]
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

#         M = Q_K_sample.max(-1)[0] - torch.mean(Q_K_sample, dim=-1)
#         M_top = M.topk(actual_n_top, sorted=False)[1]  # 使用修正后的n_top
        
#         return M_top

#     def _get_initial_context(self, V, L_Q):
#         B, H, L_V, D = V.shape
#         if not self.mask_flag:
#             V_sum = V.mean(dim=-2)
#             contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
#         else:
#             contex = V.cumsum(dim=-2)
#         return contex

#     def forward(self, queries, keys, values):
#         B, H, L_Q, D = queries.shape
#         _, _, L_K, _ = keys.shape

#         queries = queries.transpose(2, 1)
#         keys = keys.transpose(2, 1)
#         values = values.transpose(2, 1)

#         U = self.factor * math.ceil(math.log(L_K))
#         u = self._prob_QK(queries, keys, sample_k=U, n_top=U)

#         scores_top = torch.matmul(queries, keys.transpose(-2, -1))
#         if self.scale:
#             scores_top = scores_top / math.sqrt(D)
        
#         attn = torch.softmax(scores_top, dim=-1)
#         attn = self.dropout(attn)
#         context = torch.matmul(attn, values)
        
#         context = context.transpose(2, 1).contiguous()
#         return context

# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
#         super().__init__()
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values):
#         B, L, _ = queries.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, L, H, -1)
#         values = self.value_projection(values).view(B, L, H, -1)

#         out = self.inner_attention(queries, keys, values)
#         out = out.view(B, L, -1)
#         return self.out_projection(out)

# class ConvLayer(nn.Module):
#     def __init__(self, c_in, window_size=3):
#         super().__init__()
#         self.down_conv = nn.Conv1d(c_in, c_in, kernel_size=window_size, padding=(window_size-1)//2, padding_mode='replicate')
#         self.norm = nn.BatchNorm1d(c_in)
#         self.activation = nn.ELU()
#         self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         x = self.down_conv(x.transpose(1, 2))
#         x = self.norm(x)
#         x = self.activation(x)
#         x = self.max_pool(x)
#         return x.transpose(1, 2)

# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(d_model, d_ff, 1)
#         self.conv2 = nn.Conv1d(d_ff, d_model, 1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == 'relu' else F.gelu

#     def forward(self, x):
#         attn_output = self.attention(x, x, x)
#         x = self.norm1(x + attn_output)
        
#         output = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
#         output = self.dropout(self.conv2(output).transpose(1, 2))
#         return self.norm2(x + output)

# class Informer(nn.Module):
#     def __init__(self, input_size=1, d_model=512, n_heads=8, e_layers=2, d_ff=2048,
#                  dropout=0.1, output_size=1, device=torch.device('cuda')):
#         super().__init__()
#         self.d_model = d_model
#         self.embedding = nn.Linear(input_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         self.encoder = nn.ModuleList([
#             nn.Sequential(
#                 EncoderLayer(
#                     AttentionLayer(
#                         ProbAttention(False, factor=5, scale=None, attention_dropout=dropout),
#                         d_model, n_heads
#                     ),
#                     d_model, d_ff, dropout, 'gelu'
#                 ),
#                 ConvLayer(d_model)
#             ) for _ in range(e_layers)
#         ])
#         self.projection = nn.Linear(d_model, output_size)
#         self.device = device

#     # def forward(self, x):
#     #     x = self.embedding(x) * math.sqrt(self.d_model)
#     #     x = self.pos_encoder(x)
        
#     #     for layer in self.encoder:
#     #         x = layer(x)
            
#     #     output = self.projection(x[:, -1, :])
#     #     return output.unsqueeze(1)
#     def forward(self, x):
#         x = self.embedding(x) * math.sqrt(self.d_model)
#         x = self.pos_encoder(x)
        
#         for layer in self.encoder:
#             x = layer(x)
            
#         # 修改这里 - 输出整个序列而非最后一个时间步
#         output = self.projection(x)  # 输出形状: [batch, seq_len, output_size]
#         return output
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape
        
        # 确保n_top不超过序列长度
        actual_n_top = min(n_top, L_Q)
        
        K_expand = K.unsqueeze(2).expand(-1, -1, L_Q, -1, -1)
        
        # 确保sample_k不超过L_K
        actual_sample_k = min(sample_k, L_K)
        index_sample = torch.randint(L_K, (L_Q, actual_sample_k)).to(Q.device)
        
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1).to(Q.device), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.mean(Q_K_sample, dim=-1)
        M_top = M.topk(actual_n_top, sorted=False)[1]
        
        return M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1])
        else:
            contex = V.cumsum(dim=-2)
        return contex

    def forward(self, queries, keys, values):
        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U = self.factor * math.ceil(math.log(L_K))
        u = self._prob_QK(queries, keys, sample_k=U, n_top=U)

        scores_top = torch.matmul(queries, keys.transpose(-2, -1))
        if self.scale:
            scores_top = scores_top / math.sqrt(D)
        
        attn = torch.softmax(scores_top, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, values)
        
        context = context.transpose(2, 1).contiguous()
        return context

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, L, H, -1)
        values = self.value_projection(values).view(B, L, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size=3):
        super().__init__()
        self.down_conv = nn.Conv1d(c_in, c_in, kernel_size=window_size, 
                                  padding=(window_size-1)//2, padding_mode='replicate')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv(x.transpose(1, 2))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x.transpose(1, 2)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        output = self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        output = self.dropout(self.conv2(output).transpose(1, 2))
        return self.norm2(x + output)

class Informer(nn.Module):
    def __init__(self, input_size=1, d_model=512, n_heads=8, e_layers=2, d_ff=2048,
                 dropout=0.1, output_size=1, device=torch.device('cuda')):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder with attention and conv layers
        self.encoder = nn.ModuleList([
            nn.Sequential(
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor=5, scale=None, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout, 'gelu'
                ),
                ConvLayer(d_model)
            ) for _ in range(e_layers)
        ])
        
        # Final projection layer - outputs sequence of same length as input
        self.projection = nn.Linear(d_model, output_size)
    #     self.projection = nn.Sequential(
    # nn.Linear(d_model, d_model),
    # nn.Upsample(scale_factor=2**e_layers, mode='linear'),  # 上采样回原长度
    # nn.Linear(d_model, output_size)
# )
        self.device = device

    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_size]
        assert len(x.shape) == 3, f"Input must be 3D tensor, got {x.shape}"
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Encoder layers
        for layer in self.encoder:
            x = layer(x)
            
        # Project to output dimension
        output = self.projection(x)  # shape: [batch_size, seq_len, output_size]
        
        return output
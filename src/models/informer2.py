# model/informer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(position * self.div_term)
        pe[..., 1::2] = torch.cos(position * self.div_term)
        return pe

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode='circular'
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
class ProbMask:
    def __init__(self, B, H, L_Q, index, scores, device="cpu"):
        _mask = torch.ones(L_Q, scores.shape[-1], dtype=torch.bool).to(device)  # (L_Q, L_K)
        _mask_ex = _mask[None, None, :].expand(B, H, L_Q, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        return M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # 添加ProbMask定义后，此处可以正常工作
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -torch.inf)  # 使用torch.inf替代np.inf

        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        scores_top, index = self._prob_QK(queries, keys, sample_k=U, n_top=U)
        context = self._get_initial_context(values, L_Q)
        context = self._update_context(context, values, scores_top, index, L_Q)
        
        return context.transpose(2, 1).contiguous()

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
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        y = x.transpose(1, 2)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        
        return self.norm2(x + y), attn

class Informer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pred_len = config.train_window
        self.output_size = 1
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.e_layers = config.e_layers
        self.d_ff = config.d_ff
        self.dropout = config.dropout
        self.device = config.device

        # 输入嵌入
        self.enc_embedding = DataEmbedding(1, self.d_model, self.dropout)
        
        # 编码器
        self.encoder = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, config.factor, attention_dropout=config.dropout),
                    self.d_model, self.n_heads
                ),
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation='gelu'
            ) for _ in range(self.e_layers)
        ])
        
        # 输出层
        self.projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.output_size)
        )

    def forward(self, x):
        # x形状: (batch_size, seq_len, 1)
        enc_out = self.enc_embedding(x)  # (batch_size, seq_len, d_model)
        
        # 编码器处理
        for layer in self.encoder:
            enc_out, _ = layer(enc_out)
        
        # 取最后时间步
        dec_out = enc_out[:, -1, :]
        
        # 投影输出
        return self.projection(dec_out).unsqueeze(-1)  # (batch_size, 1, 1)
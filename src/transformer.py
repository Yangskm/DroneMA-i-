# model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1, 0, 2)  # 恢复 [batch, seq, features]

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=8, 
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入嵌入层
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 确保与PyTorch 1.12+兼容
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, 1)
        
        # 指数层（保持与其他模型一致）
        self.exp_layer = LearnableExponentialLayer()

    def forward(self, x):
        # x shape: [batch_size, seq_len, 1]
        x = self.embedding(x)  # [batch, seq, d_model]
        x = self.pos_encoder(x)  # [batch, seq, d_model]
        
        # 调整维度为PyTorch Transformer需要的格式
        x = x.permute(1, 0, 2)  # [seq, batch, d_model]
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [seq, batch, d_model]
        
        # 恢复维度
        x = x.permute(1, 0, 2)  # [batch, seq, d_model]
        
        # 输出预测
        output = self.output_layer(x)  # [batch, seq, 1]
        return self.exp_layer(output)

class LearnableExponentialLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.exp(x * self.scale + self.bias)
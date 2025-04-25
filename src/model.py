import torch.nn as nn
import torch
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, nhead=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scale the embeddings
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x
class ExponentialLayer(nn.Module):
    def __init__(self):
        super(ExponentialLayer, self).__init__()

    def forward(self, x):
        return torch.exp(x * torch.log(torch.tensor(10.0)))

class R2DTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type='transformer'):
        super().__init__()
        
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == 'transformer':
            self.rnn = TransformerEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                nhead=4,
                dropout=0.1
            )
            
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            ExponentialLayer(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        out = self.rnn(x)
        return self.fc(out)
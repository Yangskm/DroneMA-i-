import torch

class Config:
    seed = 10086
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 60
    train_window = 60
    model_type = 'transformer'
    batch_size = 128
    learning_rate = 0.001
    model_save_path = './saved_models/'
    results_save_path = './results/'
     # Transformer专用参数
    nhead = 2# 注意力头数
    num_layers = 3 # Transformer编码器层数
    dim_feedforward = 16 # 前馈网络维度
    dropout = 0.4# Dropout率
config = Config()

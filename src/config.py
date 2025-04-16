import torch

class Config:
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10
    train_window = 70
    # model_type = 'gru'
    model_type = 'informer'  # 修改模型类型
   
    batch_size = 128
    learning_rate = 0.001
    model_save_path = './saved_models/'
    results_save_path = './results/'
#  # Informer专用参数
#     d_model = 512
#     n_heads = 8
#     e_layers = 2
#     d_ff = 2048
#     dropout = 0.1
  # Informer2专用参数
    factor = 5           # ProbSparse注意力因子
    d_model = 512        # 模型维度
    n_heads = 8          # 注意力头数
    e_layers = 2         # 编码器层数
    d_ff = 2048          # 前馈网络维度
    dropout = 0.1        # Dropout率
    activation = 'gelu'  # 激活函数类型
config = Config()

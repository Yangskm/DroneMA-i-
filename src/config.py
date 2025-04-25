import torch

class Config:
    seed = 10086
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 60
    train_window = 60
    model_type = 'informer'
    batch_size = 128
    learning_rate = 0.001
    model_save_path = './saved_models/'
    results_save_path = './results/'

config = Config()

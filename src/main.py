from config import config
from data_processing import create_dataloader
from model import R2DGRU
from utils import seed_everything, get_files, create_dirs
from train import train_model
from models.informer import Informer

if __name__ == "__main__":
    # 初始化设置
    seed_everything(config.seed)
    create_dirs(config.model_save_path, config.results_save_path)
    
    # 获取数据
    train_files = get_files("C:/Users/Le'novo/Desktop/毕设/DroneMA/data/positive")
    test_files = get_files("C:/Users/Le'novo/Desktop/毕设/DroneMA/data/negtive")
    print(f"找到的训练文件数量: {len(train_files)}")
    print(f"找到的测试文件数量: {len(test_files)}")
    # 创建数据加载器
    train_loader = create_dataloader(train_files, config.train_window, config.batch_size)
    test_loader = create_dataloader(test_files, config.train_window, config.batch_size)
    
    # 初始化模型
    # model = R2DGRU(input_size=1, 
    #                   hidden_size=32, 
    #                   output_size=1, 
    #                   model_type=config.model_type)
    model = Informer(
    input_size=1,
    d_model=config.d_model,
    n_heads=config.n_heads,
    e_layers=config.e_layers,
    d_ff=config.d_ff,
    dropout=config.dropout,
    output_size=1,
    device=config.device
)

    
    # 训练模型
    trained_model = train_model(model, train_loader, test_loader, config)

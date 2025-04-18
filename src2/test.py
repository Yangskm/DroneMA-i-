# # test.py
# import torch
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr, spearmanr # Added for correlation
# from utils import normalize # Assuming normalize is in utils.py

# # Placeholder for the missing function - REMOVE or IMPLEMENT this
# # def apply_exponential_weights(series, alpha):
# #     # Implement your exponential weighting logic here
# #     print("Warning: apply_exponential_weights is not implemented!")
# #     return series # Return original series as fallback

# def test_model(model, loader, device, loss_function, save_path='results.csv', ew_alpha=0):
#     """
#     Runs the model on the data from the loader, calculates metrics per batch,
#     and saves detailed results per sample.

#     Args:
#         model: The trained PyTorch model.
#         loader: DataLoader providing test data.
#         device: The device to run the model on ('cuda' or 'cpu').
#         loss_function: The loss function used for evaluation.
#         save_path (str): Path to save the resulting CSV file.
#         ew_alpha (float): Alpha for exponential weighting in correlation (0 to disable).

#     Output CSV Columns:
#         - Batch_Index: Index of the batch the sample belonged to.
#         - Loss: Loss calculated for the batch (duplicated for samples in the same batch).
#         - pearson: Pearson correlation calculated for the batch (duplicated).
#         - spearman: Spearman correlation calculated for the batch (duplicated).
#         - input: Last element of the raw input sequence for the sample.
#         - target: Last element of the raw target sequence for the sample.
#         - input_norm: Last element of the normalized input sequence for the sample.
#         - target_norm: Last element of the normalized target sequence for the sample.
#         - output: Last element of the model's output sequence for the sample.
#         (Column names adapted slightly from original test function for clarity)
#     """
#     model.eval()
#     results_list = [] # Store dictionaries, one per sample

#     print(f"  Running test model, calculating metrics, saving results to {save_path}")
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             batch_size = inputs.size(0)

#             # --- Normalization ---
#             try:
#                 # Normalize inputs (assuming model expects normalized)
#                 inputs_normalized = normalize(inputs)
#                 # Normalize targets (used for loss and correlation as per old code)
#                 targets_normalized = normalize(targets)
#             except Exception as e:
#                  print(f"Error during normalization in batch {batch_idx}: {e}")
#                  # Handle error - skip batch or use raw values? Skipping for safety.
#                  print(f"    Skipping batch {batch_idx} due to normalization error.")
#                  continue

#             # --- Model Prediction ---
#             outputs = model(inputs_normalized) # Use normalized inputs

#             # --- Loss Calculation (using last elements) ---
#             try:
#                 # Ensure dimensions are suitable for loss (e.g., [batch, features])
#                 last_output = outputs[:, -1, :]
#                 last_target_norm = targets_normalized[:, -1, :]
#                 loss = loss_function(last_output, last_target_norm).item()
#             except Exception as e:
#                  print(f"Error calculating loss in batch {batch_idx}: {e}")
#                  loss = np.nan # Assign NaN if loss calculation fails

#             # --- Correlation Calculation (using full sequences) ---
#             pearson_correlation = np.nan
#             spearman_correlation = np.nan
#             try:
#                 # Get full sequences as flattened numpy arrays
#                 # Ensure outputs and targets_normalized have compatible shapes for flattening
#                 predicted_flat = outputs.cpu().numpy().flatten()
#                 real_flat = targets_normalized.cpu().numpy().flatten()

#                 # Check if arrays are suitable for correlation
#                 if len(predicted_flat) > 1 and len(real_flat) > 1 and len(predicted_flat) == len(real_flat) and \
#                    np.std(predicted_flat) > 1e-6 and np.std(real_flat) > 1e-6: # Avoid constant arrays

#                     if ew_alpha != 0:
#                         # --- Exponential Weighting (Requires apply_exponential_weights) ---
#                         predicted_series = pd.Series(predicted_flat)
#                         real_series = pd.Series(real_flat)
#                         alpha = ew_alpha
#                         print(f"    Applying exponential weighting with alpha={alpha} (Function needs implementation!)")
#                         # ==== !!! Function 'apply_exponential_weights' is NOT DEFINED !!! ====
#                         # predicted_weighted = apply_exponential_weights(predicted_series, alpha)
#                         # real_weighted = apply_exponential_weights(real_series, alpha)
#                         # pearson_correlation = predicted_weighted.corr(real_weighted, method='pearson')
#                         # spearman_correlation = predicted_weighted.corr(real_weighted, method='spearman')
#                         # =====================================================================
#                         print("    Skipping EW correlation as apply_exponential_weights is missing.")

#                     else:
#                         # --- Standard Correlation ---
#                         pearson_correlation = pearsonr(predicted_flat, real_flat).statistic
#                         spearman_correlation = spearmanr(predicted_flat, real_flat).statistic
#                 else:
#                      print(f"    Skipping correlation calculation for batch {batch_idx}: Invalid data shape or variance.")

#             except Exception as e:
#                 print(f"Error calculating correlation in batch {batch_idx}: {e}")
#                 # Keep correlations as NaN

#             # --- Collect results PER SAMPLE ---
#             # Store batch-level metrics and per-sample last elements
#             for i in range(batch_size):
#                  sample_data = {
#                      'Batch_Index': batch_idx,
#                      'Loss': loss, # Batch-level metric
#                      'pearson': pearson_correlation, # Batch-level metric
#                      'spearman': spearman_correlation, # Batch-level metric
#                      # Per-sample last elements (assuming single feature)
#                      'input': inputs[i, -1, 0].item() if inputs.shape[-1] == 1 else inputs[i, -1, :].cpu().numpy(),
#                      'target': targets[i, -1, 0].item() if targets.shape[-1] == 1 else targets[i, -1, :].cpu().numpy(),
#                      'input_norm': inputs_normalized[i, -1, 0].item() if inputs_normalized.shape[-1] == 1 else inputs_normalized[i, -1, :].cpu().numpy(),
#                      'target_norm': targets_normalized[i, -1, 0].item() if targets_normalized.shape[-1] == 1 else targets_normalized[i, -1, :].cpu().numpy(),
#                      'output': outputs[i, -1, 0].item() if outputs.shape[-1] == 1 else outputs[i, -1, :].cpu().numpy(),
#                  }
#                  # Handle multi-feature case by storing arrays or multiple columns if needed
#                  # Example for multi-feature (less ideal for standard CSV):
#                  # 'input_last': inputs[i, -1, :].cpu().numpy(),
#                  # ... etc
#                  results_list.append(sample_data)

#     # --- Save collected results ---
#     if not results_list:
#          print("Warning: No results collected during testing. Output CSV will be empty or not generated.")
#          # Create empty DataFrame with expected columns
#          results_df = pd.DataFrame(columns=['Batch_Index', 'Loss', 'pearson', 'spearman',
#                                             'input', 'target', 'input_norm', 'target_norm', 'output'])
#     else:
#         results_df = pd.DataFrame(results_list)
#         # Define column order if desired
#         cols = ['Batch_Index', 'Loss', 'pearson', 'spearman',
#                 'input', 'target', 'input_norm', 'target_norm', 'output']
#         results_df = results_df[cols] # Reorder


#     try:
#         results_df.to_csv(save_path, index=False, float_format='%.6f') # Use float format
#         print(f"  Test results successfully saved to {save_path}")
#     except Exception as e:
#         print(f"Error saving test results to {save_path}: {e}")
# test.py
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from utils import normalize

def apply_exponential_weights(series, alpha):
    """
    对序列应用指数加权
    Args:
        series: 输入序列 (numpy数组或pandas Series)
        alpha: 加权系数 (0-1)
    Returns:
        加权后的序列 (numpy数组)
    """
    if alpha <= 0 or alpha >= 1:
        return series  # 无效的alpha值，返回原始序列
    
    weights = np.array([(1 - alpha) ** i for i in range(len(series))][::-1])
    weighted_series = series * weights
    return weighted_series / weights.sum()  # 归一化

def test_model(model, loader, device, loss_function, save_path='results.csv', ew_alpha=0):
    """
    运行模型测试并保存结果
    Args:
        model: 训练好的PyTorch模型
        loader: 测试数据DataLoader
        device: 运行设备 ('cuda'或'cpu')
        loss_function: 损失函数
        save_path: 结果保存路径
        ew_alpha: 指数加权系数 (0表示禁用)
    """
    model.eval()
    results_list = []
    print(f"Running test model, saving results to {save_path}")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 归一化处理
            try:
                inputs_normalized = normalize(inputs)
                targets_normalized = normalize(targets)
            except Exception as e:
                print(f"Normalization error in batch {batch_idx}: {e}")
                continue

            # 模型预测
            outputs = model(inputs_normalized)

            # 计算损失
            try:
                loss = loss_function(outputs[:, -1, :], targets_normalized[:, -1, :]).item()
            except Exception as e:
                print(f"Loss calculation error in batch {batch_idx}: {e}")
                loss = np.nan

            # 为每个样本计算指标
            for i in range(batch_size):
                # 获取当前样本的完整序列
                output_seq = outputs[i, :, 0].cpu().numpy()
                output_len = outputs.shape[1]
                target_seq = targets_normalized[i, :output_len, 0].cpu().numpy()  # 裁剪到与输出相同长度
                # target_seq = targets_normalized[i, :, 0].cpu().numpy()

                print(f"Sample {i} (Batch {batch_idx}):")
                print(f"  Output sequence: {output_seq}")
                print(f"  Target sequence: {target_seq}")

                pearson = np.nan
                spearman = np.nan
                
                if len(output_seq) > 1 and len(target_seq) > 1:
                    try:
                        if ew_alpha > 0:
                            # 应用指数加权
                            print(f"  Applying exponential weights with alpha={ew_alpha}")
                            output_weighted = apply_exponential_weights(output_seq, ew_alpha)
                            target_weighted = apply_exponential_weights(target_seq, ew_alpha)
                            pearson, _ = pearsonr(output_weighted, target_weighted)
                            spearman, _ = spearmanr(output_weighted, target_weighted)
                        else:
                            # 标准相关性计算
                            pearson, _ = pearsonr(output_seq, target_seq)
                            spearman, _ = spearmanr(output_seq, target_seq)

                        print(f"  Pearson correlation: {pearson}")
                        print(f"  Spearman correlation: {spearman}")

                        # 如果pearson或spearman为NaN，给定默认值
                        if np.isnan(pearson):
                            print(f"  Pearson is NaN for sample {i}, setting to 0")
                            pearson = 0
                        if np.isnan(spearman):
                            print(f"  Spearman is NaN for sample {i}, setting to 0")
                            spearman = 0

                    except Exception as ex:
                        print(f"  Error calculating correlation for batch {batch_idx}, sample {i}: {ex}")
                        pearson = 0  # 默认值
                        spearman = 0  # 默认值
                else:
                    print(f"  Skipping sample {i} due to short sequence length.")
                    pearson = 0
                    spearman = 0

                # 收集结果
                sample_data = {
                    'Batch_Index': batch_idx,
                    'Loss': loss,
                    'pearson': pearson,
                    'spearman': spearman,
                    'input': inputs[i, -1, 0].item(),
                    'target': targets[i, -1, 0].item(),
                    'input_norm': inputs_normalized[i, -1, 0].item(),
                    'target_norm': targets_normalized[i, -1, 0].item(),
                    'output': outputs[i, -1, 0].item(),
                    'label': 1 if 'positive' in str(loader.dataset.processed_data[i][0]) else 0
                }
                results_list.append(sample_data)

    # 保存结果
    if results_list:
        results_df = pd.DataFrame(results_list)
        cols = ['Batch_Index', 'Loss', 'pearson', 'spearman', 'input', 'target', 
                'input_norm', 'target_norm', 'output', 'label']
        results_df = results_df[cols]
        try:
            results_df.to_csv(save_path, index=False, float_format='%.6f')
            print(f"Test results saved to {save_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    else:
        print("Warning: No results collected. Empty CSV will be generated.")
        results_df = pd.DataFrame(columns=cols)
    
    return results_df

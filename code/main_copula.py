import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import torch.nn as nn

# 导入自定义模块
from models import MultiViewFusionModel
from dataset import myDataSet
from utils_copula import (
    LogCoshLoss, parametricLoss, calculate_sigma_hat,
    train_model_copula, test_stage_copula, plot_losses)
   
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 设置随机种子
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_state = 567
    set_seed(random_state)
    
    # 配置参数
    label_columns = ['TLC实测值', 'FVC实测值', 'FEV1实测值']
    target_size = (224, 224)
    batch_size = 32
    num_epochs = 100
    num_slices_per_view = 10
    test_ratio = 0.1
    
    # 数据预处理
    ann_file1 = './modified_data_1222.csv'
    ann_file2 = './modified_data_863.csv'
    ann_file3 = './modified_data_975.csv'
    
    labels1 = pd.read_csv(ann_file1, dtype={'new_ID': str})
    labels2 = pd.read_csv(ann_file2, dtype={'new_ID': str})
    labels3 = pd.read_csv(ann_file3, dtype={'new_ID': str})
    
    # 合并患者ID
    combined_ids = list({**{pid: None for pid in labels1['new_ID']}, 
                         **{pid: None for pid in labels2['new_ID']},
                         **{pid: None for pid in labels3['new_ID']}})
    
    print(f"合并后总患者数（去重后）: {len(combined_ids)}")
    
    # 移除无效ID
    invalid_ids_to_remove = {
        "0022439420-20230703", "0003277197-20230804", "0022553264-20230727", 
        "0000433566-20221101", "0022405018-20230512", "0021883820-20230302",
        # ... 其他无效ID
    }
    
    # 筛选有效患者
    label_dfs = [labels1, labels2, labels3]
    valid_ids = []
    for pid in combined_ids:
        if pid in invalid_ids_to_remove:
            continue
        
        has_valid_label = False
        for df in label_dfs:
            if pid in df['new_ID'].values:
                label = df[df['new_ID'] == pid][label_columns].values
                if label.size == len(label_columns) and not np.isnan(label).any():
                    has_valid_label = True
                    break
        
        if has_valid_label:
            valid_ids.append(pid)
    
    print(f"有效患者数: {len(valid_ids)}")
    
    # 创建分层依据
    pid_to_fev1 = {}
    pid_to_fvc = {}
    for pid in valid_ids:
        for df in label_dfs:
            if pid in df['new_ID'].values:
                pid_to_fev1[pid] = df[df['new_ID']==pid]['FEV1实测值'].values[0]
                pid_to_fvc[pid] = df[df['new_ID']==pid]['FVC实测值'].values[0]
                break
    
    fev1_series = pd.Series(list(pid_to_fev1.values()))
    fvc_series = pd.Series(list(pid_to_fvc.values()))
    
    fev1_qcut = pd.qcut(fev1_series, 3, labels=False).astype(int)
    fvc_qcut = pd.qcut(fvc_series, 3, labels=False).astype(int)
    
    strata_labels = [
        (int(fev1_qcut[i]), int(fvc_qcut[i]))
        for i in range(len(valid_ids))
    ]
    
    all_strata = [(i, j) for i in range(3) for j in range(3)]
    strata_to_idx = {s: i for i, s in enumerate(all_strata)}
    strata_indices = np.array([strata_to_idx[s] for s in strata_labels])
    
    # 划分训练验证集和测试集
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    for train_val_idx, test_idx in sss_test.split(valid_ids, strata_indices):
        train_val_pids = [valid_ids[i] for i in train_val_idx]
        test_pids = [valid_ids[i] for i in test_idx]
        train_val_strata = strata_indices[train_val_idx]
    
    print(f"测试集大小: {len(test_pids)}, 占比: {len(test_pids)/len(valid_ids):.2%}")
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    all_results = {
        'mse': [], 'rmse': [], 'mae': [], 
        'log_cosh': [], 'copula_loss': [],
        'train_copula': [], 'val_copula': [],
        'train_log_cosh': [], 'val_log_cosh': []
    }
    all_training_times = []
    
    skipped_folds = [0, 1, 2]  # 可以跳过的fold
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_pids, train_val_strata)):
        print(f"\n{'='*50}")
        print(f"开始第 {fold+1}/5 折交叉验证（Copula训练）")
        print(f"{'='*50}")
        
        if fold in skipped_folds:
            print(f"跳过第{fold+1}折")
            continue
        
        # 获取当前fold的数据
        fold_train_pids = [train_val_pids[i] for i in train_idx]
        fold_val_pids = [train_val_pids[i] for i in val_idx]
        
        print(f"第{fold+1}折 - 训练集: {len(fold_train_pids)}, 验证集: {len(fold_val_pids)}")
        
        # 标准化标签
        train_labels_matrix = []
        for pid in fold_train_pids:
            for df in label_dfs:
                if pid in df['new_ID'].values:
                    label = df[df['new_ID']==pid][label_columns].values[0]
                    train_labels_matrix.append(label)
                    break
        
        train_labels_matrix = np.array(train_labels_matrix)
        scaler = StandardScaler()
        scaler.fit(train_labels_matrix)
        
        # 创建数据集
        train_dataset = myDataSet(
            idx_list=fold_train_pids,
            if_augmentation=True,
            annotations_files=[ann_file1, ann_file2, ann_file3],
            parent_dirs=["./CT_Images/ROI_1222/", "./CT_Images/ROI_863/", "./CT_Images/ROI_975/"],
            target_size=target_size,
            scaler=scaler,
            label_columns=label_columns,
            aug_factor=3,
            num_slices_per_view=num_slices_per_view
        )
        
        val_dataset = myDataSet(
            idx_list=fold_val_pids,
            if_augmentation=False,
            annotations_files=[ann_file1, ann_file2, ann_file3],
            parent_dirs=["./CT_Images/ROI_1222/", "./CT_Images/ROI_863/", "./CT_Images/ROI_975/"],
            target_size=target_size,
            scaler=scaler,
            label_columns=label_columns,
            aug_factor=1,
            num_slices_per_view=num_slices_per_view
        )
        
        test_dataset = myDataSet(
            idx_list=test_pids,
            if_augmentation=False,
            annotations_files=[ann_file1, ann_file2, ann_file3],
            parent_dirs=["./CT_Images/ROI_1222/", "./CT_Images/ROI_863/", "./CT_Images/ROI_975/"],
            target_size=target_size,
            scaler=scaler,
            label_columns=label_columns,
            aug_factor=1,
            num_slices_per_view=num_slices_per_view
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # ==================== 第一阶段：加载预训练模型 ====================
        print(f"\n第{fold+1}折 - 第一阶段：加载预训练模型")
        net = MultiViewFusionModel(num_slices=num_slices_per_view).to(device)
        
        # 加载第一阶段训练的模型权重
        first_stage_model_path = f"./logcos/模型权重_seed{random_state}_fold{fold}.pth"
        try:
            state_dict = torch.load(first_stage_model_path, map_location=device)
            net.load_state_dict(state_dict)
            print(f"成功加载第一阶段模型权重: {first_stage_model_path}")
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            print("将使用随机初始化模型")
        
        # ==================== 第二阶段：Copula训练 ====================
        print(f"\n第{fold+1}折 - 第二阶段：Copula训练")
        
        # 计算固定sigma_hat
        print("计算固定协方差矩阵sigma_hat...")
        sigma_hat = calculate_sigma_hat(train_loader, net, device)
        print(f"sigma_hat:\n{sigma_hat}")
        
        # 创建Copula损失
        copula_criterion = parametricLoss()
        
        # 优化器和调度器
        optimizer = optim.SGD(
            net.parameters(), 
            lr=0.01, 
            weight_decay=1e-3, 
            momentum=0.9
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        # 训练模型（使用Copula损失）
        start_time = time.time()
        train_copula, val_copula, train_log_cosh, val_log_cosh, trained_model = train_model_copula(
            model=net,
            criterion=copula_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dl_train=train_loader,
            dl_val=val_loader,
            device=device,
            num_epochs=num_epochs,
            use_copula=True,
            copula_criterion=copula_criterion,
            sigma_hat=sigma_hat
        )
        training_time = time.time() - start_time
        
        # 保存结果
        all_training_times.append(training_time)
        all_results['train_copula'].append(train_copula)
        all_results['val_copula'].append(val_copula)
        all_results['train_log_cosh'].append(train_log_cosh)
        all_results['val_log_cosh'].append(val_log_cosh)
        
        # 保存模型
        model_save_path = f"./logcos/模型权重_seed{random_state}_fold{fold}_copula.pth"
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"模型已保存到: {model_save_path}")
        
        # 测试模型
        print(f"\n第{fold+1}折 - 测试阶段")
        mse, rmse, mae, copula_loss, log_cosh = test_stage_copula(
            trained_model, test_loader, scaler, device,
            use_copula=True,
            copula_criterion=copula_criterion,
            sigma_hat=sigma_hat
        )
        
        all_results['mse'].append(mse)
        all_results['rmse'].append(rmse)
        all_results['mae'].append(mae)
        all_results['log_cosh'].append(log_cosh)
        all_results['copula_loss'].append(copula_loss)
        
        print(f"\n第{fold+1}折训练完成，耗时: {training_time:.2f}秒")
        print(f"测试指标 - Log Cosh: {log_cosh:.4f}")
        print(f"          MSE: {mse[0]:.4f}, {mse[1]:.4f}, {mse[2]:.4f}")
        print(f"          RMSE: {rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f}")
        print(f"          MAE: {mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f}")
    
    # 汇总结果
    print(f"\n{'='*50}")
    print("五折交叉验证结果汇总 (Copula训练)")
    print(f"{'='*50}")
    
    # 计算平均指标
    mse_array = np.array(all_results['mse'])
    rmse_array = np.array(all_results['rmse'])
    mae_array = np.array(all_results['mae'])
    log_cosh_array = np.array(all_results['log_cosh'])
    
    mean_mse = np.mean(mse_array, axis=0)
    std_mse = np.std(mse_array, axis=0)
    mean_rmse = np.mean(rmse_array, axis=0)
    std_rmse = np.std(rmse_array, axis=0)
    mean_mae = np.mean(mae_array, axis=0)
    std_mae = np.std(mae_array, axis=0)
    mean_log_cosh = np.mean(log_cosh_array)
    std_log_cosh = np.std(log_cosh_array)
    
    print(f"平均测试指标 - Log Cosh: {mean_log_cosh:.4f}±{std_log_cosh:.4f}")
    print(f"              MSE: {mean_mse[0]:.4f}±{std_mse[0]:.4f}, {mean_mse[1]:.4f}±{std_mse[1]:.4f}, {mean_mse[2]:.4f}±{std_mse[2]:.4f}")
    print(f"              RMSE: {mean_rmse[0]:.4f}±{std_rmse[0]:.4f}, {mean_rmse[1]:.4f}±{std_rmse[1]:.4f}, {mean_rmse[2]:.4f}±{std_rmse[2]:.4f}")
    print(f"              MAE: {mean_mae[0]:.4f}±{std_mae[0]:.4f}, {mean_mae[1]:.4f}±{std_mae[1]:.4f}, {mean_mae[2]:.4f}±{std_mae[2]:.4f}")
    
    avg_training_time = np.mean(all_training_times)
    print(f"平均训练时间: {avg_training_time:.2f}秒/折")
    
    # 保存结果到文件
    results_df = pd.DataFrame({
        'Fold': range(1, len(all_results['mse']) + 1),
        'LogCosh': all_results['log_cosh'],
        'MSE_TLC': [mse[0] for mse in all_results['mse']],
        'MSE_FVC': [mse[1] for mse in all_results['mse']],
        'MSE_FEV1': [mse[2] for mse in all_results['mse']],
        'MAE_TLC': [mae[0] for mae in all_results['mae']],
        'MAE_FVC': [mae[1] for mae in all_results['mae']],
        'MAE_FEV1': [mae[2] for mae in all_results['mae']],
        'Training_Time': all_training_times
    })
    
    results_df.to_csv('copula_logcosh_cross_validation_results.csv', index=False)
    print("结果已保存到 copula_logcosh_cross_validation_results.csv")

if __name__ == "__main__":
    main()
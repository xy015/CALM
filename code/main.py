import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

# 导入自定义模块
from models import MultiViewFusionModel
from dataset import myDataSet
from utils import train_model, test_stage, plot_loss_curves, LogCoshLoss

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_state = 567
    set_seed(random_state)
    
    label_columns = ['TLC实测值', 'FVC实测值', 'FEV1实测值']
    target_size = (224, 224)
    batch_size = 32
    num_epochs = 100
    num_slices_per_view = 10
    test_ratio = 0.1  # 测试集占比10%
    val_ratio = 0.2  # 验证集占比20%（基于总数据）
    
    # 数据预处理 
    ann_file1 = './modified_data_1222.csv'
    ann_file2 = './modified_data_863.csv'
    ann_file3 = './modified_data_975.csv'
    labels1 = pd.read_csv(ann_file1, dtype={'new_ID': str})
    labels2 = pd.read_csv(ann_file2, dtype={'new_ID': str})
    labels3 = pd.read_csv(ann_file3, dtype={'new_ID': str})
    combined_ids = list({**{pid: None for pid in labels1['new_ID']}, **{pid: None for pid in labels2['new_ID']},**{pid: None for pid in labels3['new_ID']}})
    print(f"合并后总患者数（去重后）: {len(combined_ids)}")

    invalid_ids_to_remove = {
        "0022439420-20230703", "0003277197-20230804", "0022553264-20230727", "0000433566-20221101"
    }

    def is_valid_label(pid, label_dfs, label_cols):
        for df in label_dfs:
            if pid in df['new_ID'].values:
                label = df[df['new_ID'] == pid][label_cols].values
                if label.size == len(label_cols) and not np.isnan(label).any():
                    return True
        return False

    label_dfs = [labels1,labels2,labels3]
    valid_ids = [pid for pid in combined_ids if pid not in invalid_ids_to_remove and is_valid_label(pid, label_dfs, label_columns)]
    print(f"有效患者数: {len(valid_ids)}")

    # 分层依据（FEV1和FVC）
    pid_to_fev1 = {pid: df[df['new_ID']==pid]['FEV1实测值'].values[0] for pid in valid_ids for df in label_dfs if pid     in df['new_ID'].values}
    pid_to_fvc = {pid: df[df['new_ID']==pid]['FVC实测值'].values[0] for pid in valid_ids for df in label_dfs if pid       in df['new_ID'].values}

    fev1_series = pd.Series(pid_to_fev1.values())
    fvc_series = pd.Series(pid_to_fvc.values())
    fev1_qcut = pd.qcut(fev1_series, 3, labels=False).astype(int) 
    fvc_qcut = pd.qcut(fvc_series, 3, labels=False).astype(int)   

# 生成strata_labels，确保每个元素是元组且包含Python整数
    strata_labels = [
        (int(fev1_qcut[i]), int(fvc_qcut[i]))
        for i in range(len(valid_ids))
    ]
    #strata_labels = np.array(strata_labels, dtype=object)  # 转换为object数组，元素为元组
    print(type(strata_labels))                  
    print(type(strata_labels[0]), strata_labels[0])  # 
# 生成所有可能的分层组合并建立索引映射
    all_strata = [(i, j) for i in range(3) for j in range(3)]
    strata_to_idx = {s: i for i, s in enumerate(all_strata)}

# 调试：检查是否有不在all_strata中的分层组合
    invalid_strata = [s for s in strata_labels if s not in strata_to_idx]
    if invalid_strata:
        print(f"警告：发现{len(invalid_strata)}个无效分层组合: {invalid_strata[:5]}")

# 获取每个样本的分层索引
    strata_indices = np.array([strata_to_idx[s] for s in strata_labels])

    # 划分训练集和测试集（测试集占10%）
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    for train_val_idx, test_idx in sss_test.split(valid_ids, strata_indices):
        train_val_pids = [valid_ids[i] for i in train_val_idx]
        test_pids = [valid_ids[i] for i in test_idx]
    
    print(f"测试集大小: {len(test_pids)}, 占比: {len(test_pids)/len(valid_ids):.2%}")
    
    # 训练集进行5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    all_mse = []
    all_rmse = []
    all_mae = []
    all_train_losses = []
    all_val_losses = []
    all_training_times = []
    
    # 生成训练集的分层标签（用于5折交叉验证）
    train_val_strata_indices = strata_indices[train_val_idx]
    skipped_folds=[0,1,2]
    # 开始5折循环
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_pids, train_val_strata_indices)):
        print(f"\n{'='*50}")
        print(f"开始第 {fold+1}/5 折交叉验证（训练集内划分验证集）")
        print(f"{'='*50}")
        
        # 划分训练/验证集 
        train_pids = [train_val_pids[i] for i in train_idx]
        val_pids = [train_val_pids[i] for i in val_idx]
        print(f"第{fold+1}折 - 训练集大小: {len(train_pids)}, 验证集大小: {len(val_pids)}, 测试集大小: {len(test_pids)}")
        print(f"比例 - 训练集: {len(train_pids)/len(valid_ids):.2%}, 验证集: {len(val_pids)/len(valid_ids):.2%}, 测试集: {len(test_pids)/len(valid_ids):.2%}")
        if fold in skipped_folds:
            print(f"跳过第{fold+1}折 (fold={fold})，已运行过或无需训练")
            continue
        
        # 数据标准化
        train_labels_matrix = np.array([df[df['new_ID']==pid][label_columns].values[0] for pid in train_pids for df in label_dfs if pid in df['new_ID'].values])
        scaler = StandardScaler()
        scaler.fit(train_labels_matrix)
        
        # 创建数据集 
        train_dataset = myDataSet(
            train_pids, 
            if_augmentation=True, 
            annotations_files=[ann_file1, ann_file2,ann_file3],
            parent_dirs=["./CT_Images/ROI_1222/","./CT_Images/ROI_863/","./CT_Images/ROI_975/"],
            target_size=target_size,
            scaler=scaler,
            aug_factor=3,
            num_slices_per_view=num_slices_per_view,
            cache_dir=None
        )

        val_dataset = myDataSet(
            val_pids, 
            if_augmentation=False,
            annotations_files=[ann_file1, ann_file2,ann_file3],
            parent_dirs=["./CT_Images/ROI_1222/","./CT_Images/ROI_863/","./CT_Images/ROI_975/"],
            target_size=target_size,
            scaler=scaler,
            num_slices_per_view=num_slices_per_view,
            cache_dir=None
        )

        test_dataset = myDataSet(
            test_pids, 
            if_augmentation=False,
            annotations_files=[ann_file1, ann_file2,ann_file3],
            parent_dirs=["./CT_Images/ROI_1222/","./CT_Images/ROI_863/","./CT_Images/ROI_975/"],
            target_size=target_size,
            scaler=scaler,
            num_slices_per_view=num_slices_per_view,
            cache_dir=None
        )
        
        #  数据加载器 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # 模型与优化器 
        net = MultiViewFusionModel(num_slices=num_slices_per_view).to(device)
        
        # 打印模型结构
        summary(net, input_size=(1, num_slices_per_view, 3, 224, 224))
        
        # 选择损失函数
        #criterion = nn.MSELoss()  # 选项1: MSE损失
        #criterion = nn.SmoothL1Loss(beta=0.5)  # 选项2: SmoothL1损失
        criterion = LogCoshLoss()  # 选项3: LogCosh损失（原代码使用）
        
        # 设置优化器
        optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        # 训练与评估 
        start_time = time.time()
        train_loss, val_loss, trained_model = train_model(
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dl_train=train_loader,
            dl_val=val_loader,
            dl_test=test_loader,
            device=device,
            num_epochs=num_epochs
        )
        training_time = time.time() - start_time
        all_training_times.append(training_time)
        all_train_losses.append(train_loss)
        all_val_losses.append(val_loss)
        
        # 测试与保存 
        mse, rmse, mae = test_stage(trained_model, test_loader, scaler, device)
        all_mse.append(mse)
        all_rmse.append(rmse)
        all_mae.append(mae)
        
        # 保存当前折的模型
        torch.save(trained_model.state_dict(), f'./logcos/模型权重_seed{random_state}_fold{fold}.pth')
        
        # 结果输出 
        print(f"\n第{fold+1}折训练完成，耗时: {training_time:.2f}秒")
        print(f"第{fold+1}折测试指标 - MSE: {mse[0]:.4f}, {mse[1]:.4f}, {mse[2]:.4f}")
        print(f"                RMSE: {rmse[0]:.4f}, {rmse[1]:.4f}, {rmse[2]:.4f}")
        print(f"                MAE: {mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f}")

        #  绘制当前折的损失曲线 
        def plot_current_fold_losses(train_loss, val_loss, fold):
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss, label='Train Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f' {fold+1} _5fold_mse_loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'./logcos/seed{random_state}五折交叉验证损失曲线_fold{fold+1}.png')
            plt.close()  
        
        plot_current_fold_losses(train_loss, val_loss, fold)
    # 汇总五折结果
    print(f"\n{'='*50}")
    print(f"五折交叉验证结果汇总")
    print(f"{'='*50}")
    
    # 转换为numpy数组便于计算
    all_mse = np.array(all_mse)
    all_rmse = np.array(all_rmse)
    all_mae = np.array(all_mae)
    
    # 计算平均指标和标准差
    mean_mse = np.mean(all_mse, axis=0)
    std_mse = np.std(all_mse, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    std_rmse = np.std(all_rmse, axis=0)
    mean_mae = np.mean(all_mae, axis=0)
    std_mae = np.std(all_mae, axis=0)
    
    # 输出汇总结果
    print(f"平均测试指标 - MSE: {mean_mse[0]:.4f}±{std_mse[0]:.4f}, {mean_mse[1]:.4f}±{std_mse[1]:.4f}, {mean_mse[2]:.4f}±{std_mse[2]:.4f}")
    print(f"              RMSE: {mean_rmse[0]:.4f}±{std_rmse[0]:.4f}, {mean_rmse[1]:.4f}±{std_rmse[1]:.4f}, {mean_rmse[2]:.4f}±{std_rmse[2]:.4f}")
    print(f"              MAE: {mean_mae[0]:.4f}±{std_mae[0]:.4f}, {mean_mae[1]:.4f}±{std_mae[1]:.4f}, {mean_mae[2]:.4f}±{std_mae[2]:.4f}")
    
    # 计算平均训练时间
    avg_training_time = np.mean(all_training_times)
    print(f"平均训练时间: {avg_training_time:.2f}秒/折")
    
    # 绘制各折损失曲线
    def plot_all_fold_losses(all_train_losses, all_val_losses):
        plt.figure(figsize=(12, 8))
        for fold, (train_loss, val_loss) in enumerate(zip(all_train_losses, all_val_losses)):
            plt.plot(train_loss, label=f'Fold {fold+1} Train Loss')
            plt.plot(val_loss, label=f'Fold {fold+1} Val Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('mse_loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('./logcos/mse_loss_sgd0.001_logcos.png')
        plt.show()
    
    plot_all_fold_losses(all_train_losses, all_val_losses)
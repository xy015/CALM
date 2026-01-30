import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def get_validation_loss(net, criterion, data_loader, device):
    """计算验证/测试数据的平均损失"""
    net.eval()  
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            try:
                # 解包数据并移动到设备
                inputs, labels = batch_data[0], batch_data[1]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
            except Exception as e:
                print(f"计算验证损失时出错: {e}")
                continue
    
    if total_samples == 0:
        return float('inf')
        
    return total_loss / total_samples

def train_model(model, criterion, optimizer, scheduler, dl_train, dl_val, dl_test, 
                device, num_epochs=25, log_interval=100):
    """训练函数"""
    torch.backends.cudnn.benchmark = True
    # 初始化
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0
    
    # 损失记录
    train_losses = []
    val_losses = []
    test_losses = []
    
    since = time.time()
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # ==================== 训练阶段 ====================
        model.train()
        running_loss = 0.0
        running_samples = 0
        
        for batch_idx, batch_data in enumerate(dl_train):
            try:
                inputs, labels = batch_data[0], batch_data[1]
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_samples += batch_size
                
                if batch_idx % log_interval == 0:
                    current_loss = loss.item()
                    print(f'Batch {batch_idx}/{len(dl_train)}, Loss: {current_loss:.6f}')
                    
            except Exception as e:
                print(f"训练batch {batch_idx}时出错: {e}")
                continue
        
        # 计算平均训练损失
        epoch_train_loss = running_loss / running_samples if running_samples > 0 else float('inf')
        train_losses.append(epoch_train_loss)
        
        # 验证
        val_loss = get_validation_loss(model, criterion, dl_val, device)
        val_losses.append(val_loss)
        
        # 测试
        #test_loss = get_validation_loss(model, criterion, dl_test, device)
        #test_losses.append(test_loss)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'最佳模型！验证损失: {val_loss:.6f}')
        
        # 打印epoch总结 
        current_lr = optimizer.param_groups[0]['lr']
        print(f'训练损失: {epoch_train_loss:.6f}')
        print(f'验证损失: {val_loss:.6f}')
        #print(f'测试损失: {test_loss:.6f}')
        print(f'学习率: {current_lr:.2e}')
        
        # 写入日志 
        log_entry = (f"Epoch {epoch+1:3d} | "
                    f"Train: {epoch_train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    #f"Test: {test_loss:.6f} | "
                    f"LR: {current_lr:.2e}\n")
        
        with open('training_log.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    # 训练完成
    time_elapsed = time.time() - since
    print(f'训练完成！耗时: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'最佳验证损失: {best_val_loss:.6f} (第{best_epoch+1}轮)')
    
    # 恢复最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return train_losses, val_losses, model


def plot_loss_curves(train_loss, val_loss):
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
   # plt.plot(test_loss, label="Testing Loss")
    plt.legend()
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss_curves.png')
    plt.show()

def test_stage(net, test_loader, scaler, device):
    """测试阶段函数"""
    net.eval()
    running_mae = torch.zeros(3, device=device)
    running_mse = torch.zeros(3, device=device)
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(images)
            
            # 反归一化
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            labels_inverse = scaler.inverse_transform(labels_np)
            outputs_inverse = scaler.inverse_transform(outputs_np)
            
            # 转换回GPU进行误差计算
            labels_inverse_tensor = torch.from_numpy(labels_inverse).to(device)
            outputs_inverse_tensor = torch.from_numpy(outputs_inverse).to(device)
            
            # 计算误差
            error = torch.abs(outputs_inverse_tensor - labels_inverse_tensor)
            squared_error = error ** 2
            
            # 累积误差
            running_mae += error.sum(dim=0)
            running_mse += squared_error.sum(dim=0)
            total_samples += labels_inverse.shape[0]

    # 计算平均指标
    mse = (running_mse / total_samples).cpu().numpy()
    rmse = np.sqrt(mse)
    mae = (running_mae / total_samples).cpu().numpy()

    print(f"测试 MSE: {mse}")
    print(f"测试 RMSE: {rmse}")
    print(f"测试 MAE: {mae}")
    
    return mse, rmse, mae

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # x = pred - target
        x = pred - target
        # log(cosh(x)) = x + softplus(−2x) − ln(2)
        # 这样写可以数值更稳定
        return torch.mean(x + torch.nn.functional.softplus(-2.0 * x) - torch.log(torch.tensor(2.0)))
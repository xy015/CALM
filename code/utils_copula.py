import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
from torch.distributions import multivariate_normal
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# ==================== Copula相关函数 ====================

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        x = pred - target
        return torch.mean(x + torch.nn.functional.softplus(-2.0 * x) - torch.log(torch.tensor(2.0)))

class parametricLoss(nn.Module):
    """Copula损失函数"""
    def __init__(self):
        super(parametricLoss, self).__init__()
        
    def forward(self, y_hat, y, sigma_hat):
        ei = y - y_hat
        sigma_hat = sigma_hat.to(ei.device)
        dist = multivariate_normal.MultivariateNormal(
            loc=torch.zeros(3).to(ei.device),
            covariance_matrix=sigma_hat
        )
        pdf = -dist.log_prob(ei)
        loss = torch.sum(pdf)
        return loss

def calculate_sigma_hat(dl_train, net, device):
    """计算固定协方差矩阵sigma_hat"""
    ts_resid = torch.empty(0, 3).to(device)
    net.eval()
    with torch.no_grad():
        for batch_data in dl_train:
            inputs, labels = batch_data[0], batch_data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            resid_i = labels - outputs
            ts_resid = torch.cat((ts_resid, resid_i), dim=0)
    
    ei = ts_resid
    length = ei.size(0)
    e_bar = torch.mean(ei, 0)
    ei_minus_e_bar = ei - e_bar
    sigma_i = torch.zeros(3, 3).to(device)
    
    for i in range(length):
        mat = torch.matmul(
            ei_minus_e_bar[i].reshape(3, 1),
            ei_minus_e_bar[i].reshape(1, 3)
        )
        sigma_i += mat
    sigma_hat = sigma_i / length
    sigma_hat += 1e-6 * torch.eye(3).to(device)  # 正则化
    return sigma_hat

# ==================== 训练和评估函数 ====================

def get_validation_loss(net, criterion, data_loader, device, use_copula=False, 
                       copula_criterion=None, sigma_hat=None, log_cosh_criterion=None):
    """计算验证损失 - 支持Copula loss评估"""
    net.eval()
    total_copula_loss = 0.0
    total_log_cosh = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            outputs = net(inputs)
            
            if use_copula and copula_criterion is not None and sigma_hat is not None:
                loss = copula_criterion(outputs, labels, sigma_hat) / len(inputs)
            else:
                loss = criterion(outputs, labels)
            
            if log_cosh_criterion is not None:
                log_cosh = log_cosh_criterion(outputs, labels).item()
                total_log_cosh += log_cosh * len(inputs)
            
            total_copula_loss += loss.item() * len(inputs)
            total_samples += len(inputs)
    
    return (total_copula_loss / total_samples, total_log_cosh / total_samples) if total_samples > 0 else (float('inf'), float('inf'))

def train_model_copula(model, criterion, optimizer, scheduler, dl_train, dl_val, 
                      device, num_epochs=25, log_interval=100, use_copula=False, 
                      copula_criterion=None, sigma_hat=None):
    """支持Copula损失的训练函数"""
    torch.backends.cudnn.benchmark = True
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0
    
    train_copula_losses = []
    train_log_cosh_losses = []
    val_copula_losses = []
    val_log_cosh_losses = []
    
    since = time.time()
    model = model.to(device)
    
    log_cosh_criterion = LogCoshLoss()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_copula_loss = 0.0
        running_log_cosh_loss = 0.0
        running_samples = 0
        
        for batch_idx, batch_data in enumerate(dl_train):
            try:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                if use_copula and copula_criterion is not None and sigma_hat is not None:
                    loss = copula_criterion(outputs, labels, sigma_hat) / len(inputs)
                else:
                    loss = criterion(outputs, labels)
                
                log_cosh = log_cosh_criterion(outputs, labels).item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_size = inputs.size(0)
                running_copula_loss += loss.item() * batch_size
                running_log_cosh_loss += log_cosh * batch_size
                running_samples += batch_size
                
                if batch_idx % log_interval == 0:
                    print(f'Batch {batch_idx}/{len(dl_train)}, Copula Loss: {loss.item():.6f}, Log Cosh: {log_cosh:.6f}')
                    
            except Exception as e:
                print(f"训练batch {batch_idx}时出错: {e}")
                continue
        
        epoch_train_copula_loss = running_copula_loss / running_samples if running_samples > 0 else float('inf')
        epoch_train_log_cosh_loss = running_log_cosh_loss / running_samples if running_samples > 0 else float('inf')
        train_copula_losses.append(epoch_train_copula_loss)
        train_log_cosh_losses.append(epoch_train_log_cosh_loss)
        
        val_copula_loss, val_log_cosh_loss = get_validation_loss(
            model, criterion, dl_val, device, use_copula, copula_criterion, sigma_hat, log_cosh_criterion
        )
        val_copula_losses.append(val_copula_loss)
        val_log_cosh_losses.append(val_log_cosh_loss)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_copula_loss)
        else:
            scheduler.step()
        
        if val_copula_loss < best_val_loss:
            best_val_loss = val_copula_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'新的最佳模型！验证Copula损失: {val_copula_loss:.6f}, Log Cosh: {val_log_cosh_loss:.6f}')
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'训练损失: Copula={epoch_train_copula_loss:.6f}, Log Cosh={epoch_train_log_cosh_loss:.6f}')
        print(f'验证损失: Copula={val_copula_loss:.6f}, Log Cosh={val_log_cosh_loss:.6f}')
        print(f'学习率: {current_lr:.2e}')
        
        log_entry = (f"Epoch {epoch+1:3d} | "
                    f"Train: Copula={epoch_train_copula_loss:.6f}, Log Cosh={epoch_train_log_cosh_loss:.6f} | "
                    f"Val: Copula={val_copula_loss:.6f}, Log Cosh={val_log_cosh_loss:.6f} | "
                    f"LR: {current_lr:.2e}\n")
        
        with open('training_log_copula.txt', 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    time_elapsed = time.time() - since
    print(f'\n训练完成！耗时: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'最佳验证Copula损失: {best_val_loss:.6f} (第{best_epoch+1}轮)')
    
    model.load_state_dict(best_model_wts)
    
    plot_losses(train_copula_losses, val_copula_losses, 'Copula Loss')
    plot_losses(train_log_cosh_losses, val_log_cosh_losses, 'Log Cosh Loss')
    
    return train_copula_losses, val_copula_losses, train_log_cosh_losses, val_log_cosh_losses, model

def test_stage_copula(net, test_loader, scaler, device, use_copula=False, 
                     copula_criterion=None, sigma_hat=None):
    """支持Copula损失的测试函数"""
    net.eval()
    running_mae = torch.zeros(3, device=device)
    running_mse = torch.zeros(3, device=device)
    running_log_cosh = 0.0
    total_samples = 0
    copula_loss_total = 0.0
    
    log_cosh_criterion = LogCoshLoss()
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(images)
            
            if use_copula and copula_criterion is not None and sigma_hat is not None:
                copula_loss = copula_criterion(outputs, labels, sigma_hat)
                copula_loss_total += copula_loss.item()
            
            batch_log_cosh = log_cosh_criterion(outputs, labels).item()
            running_log_cosh += batch_log_cosh * labels.size(0)
            
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            labels_inverse = scaler.inverse_transform(labels_np)
            outputs_inverse = scaler.inverse_transform(outputs_np)
            
            labels_inverse_tensor = torch.from_numpy(labels_inverse).to(device)
            outputs_inverse_tensor = torch.from_numpy(outputs_inverse).to(device)
            
            error = torch.abs(outputs_inverse_tensor - labels_inverse_tensor)
            squared_error = error ** 2
            
            running_mae += error.sum(dim=0)
            running_mse += squared_error.sum(dim=0)
            total_samples += labels_inverse.shape[0]

    mse = (running_mse / total_samples).cpu().numpy()
    rmse = np.sqrt(mse)
    mae = (running_mae / total_samples).cpu().numpy()
    log_cosh_avg = running_log_cosh / total_samples
    copula_loss_avg = copula_loss_total / total_samples if total_samples > 0 else 0.0
    
    print(f"测试 Log Cosh（标准化数据）: {log_cosh_avg:.6f}")
    print(f"测试 MSE（原始数据）: {mse}")
    print(f"测试 RMSE: {rmse}")
    print(f"测试 MAE: {mae}")
    
    if use_copula:
        print(f"测试 Copula 损失: {copula_loss_avg:.6f}")
    
    return mse, rmse, mae, copula_loss_avg, log_cosh_avg


def plot_losses(train_losses, val_losses, loss_name):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'Training {loss_name}')
    plt.plot(val_losses, label=f'Validation {loss_name}')
    plt.xlabel('Epoch')
    plt.ylabel(loss_name)
    plt.title(f'Training and Validation {loss_name}')
    plt.legend()
    plt.grid(True)
    
    filename = f'{loss_name.lower().replace(" ", "_")}_loss_curves_copula.png'
    plt.savefig(filename)
    print(f'{loss_name}曲线已保存为: {filename}')
    
    plt.close()

 
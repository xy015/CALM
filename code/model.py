import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

RADIMAGENET_WEIGHTS = "/mnt/copula肺功能/pretrain/ResNet50.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SLICES_PER_VIEW = 15  # 每个视图的切片数

# 特征提取
class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, fine_tune=True):
        super().__init__()
        # 初始化无权重的ResNet50并加载RadImageNet权重
        self.backbone = models.resnet50(weights=None)
        
        # 加载并处理RadImageNet权重
        if os.path.exists(RADIMAGENET_WEIGHTS):
            checkpoint = torch.load(RADIMAGENET_WEIGHTS, map_location=DEVICE)
            print(f"RadImageNet权重键名示例: {list(checkpoint.keys())[:5]}")
            
            # 修正权重键名以匹配原生ResNet50
            new_state_dict = {}
            for key, value in checkpoint.items():
                stripped_key = key.replace("backbone.", "")
                
                # 处理输入层映射
                if stripped_key.startswith("0."):
                    new_key = f"conv1{stripped_key[1:]}"
                elif stripped_key.startswith("1."):
                    new_key = f"bn1{stripped_key[1:]}"
                else:
                    # 处理残差块组映射
                    layer_mapping = {"4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
                    parts = stripped_key.split(".", 1)
                    if len(parts) == 2 and parts[0] in layer_mapping:
                        block_num, rest = parts
                        new_key = f"{layer_mapping[block_num]}.{rest}"
                    else:
                        new_key = stripped_key
                
                new_state_dict[new_key] = value
            
            # 非严格加载权重
            missing_keys, unexpected_keys = self.backbone.load_state_dict(
                new_state_dict, strict=False
            )
            print(f"成功加载RadImageNet权重。未加载键: {missing_keys[:5]}")
            print(f"意外键: {unexpected_keys[:5]}")
        else:
            print("未找到RadImageNet权重，使用随机初始化")
        
        # 冻结策略
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 保留除fc层外的backbone
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 替换为与原模型一致的特征提取层结构
        self.feature_linear = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x):
        # 确保输入维度正确
        if x.dim() == 3:
            x = x.unsqueeze(0)  # 添加批次维度
            
        # 特征提取流程与原模型一致
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(start_dim=1)
        return self.feature_linear(features)

# 注意力融合模块
class EnhancedCBAMFusion(nn.Module):
    def __init__(self, num_slices=15, feature_dim=64, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, feature_dim // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(feature_dim // reduction_ratio, feature_dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=5, padding=2),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Conv1d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.slice_relation = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert x.dim() == 3, f"输入维度应为3维，实际为{x.dim()}维"
        B, S, F = x.shape
        x_t = x.transpose(1, 2)  # [B, F, S]
        
        # 通道注意力
        chn_attn = self.channel_attention(x_t)  # [B, F, 1]
        x_t = x_t * chn_attn
        
        # 空间注意力
        avg_pool = torch.mean(x_t, dim=1, keepdim=True)  # [B, 1, S]
        max_pool, _ = torch.max(x_t, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, S]
        spa_attn = self.spatial_attention(spatial_input)  # [B, 1, S]
        x_t = x_t * spa_attn
        
        # 切片间关系建模
        slice_rel = self.slice_relation(x_t)  # [B, F, S]
        x_t = x_t * slice_rel
        
        return x_t.transpose(1, 2)  # [B, S, F]

# 多视图融合主模型
class MultiViewFusionModel(nn.Module):
    def __init__(self, feature_dim=32, num_classes=3, num_slices=NUM_SLICES_PER_VIEW):
        super().__init__()
        self.feature_extractor = ResNet50FeatureExtractor(feature_dim)
        self.fusion_attention = EnhancedCBAMFusion(num_slices=num_slices, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # 输入形状: [B, 20, 3, H, W]（20个3通道切片）
        B, S, C, H, W = x.shape
        assert C == 3, "输入必须为3通道（3个视图）"
        
        # 重塑为 [B*20, 3, H, W]
        x_reshaped = x.view(B * S, C, H, W)
        
        # 提取特征（输出: [B*20, feature_dim]）
        feats = self.feature_extractor(x_reshaped)
        
        # 重塑回 [B, 20, feature_dim]
        feats = feats.view(B, S, -1)
        
        # 应用注意力融合
        feats = self.fusion_attention(feats)
        
        # 调整维度为 [B, feature_dim, 20]
        feats = feats.transpose(1, 2)
        
        # 分类器处理
        logits = self.classifier(feats)  # 输出: [B, num_classes]
        return logits
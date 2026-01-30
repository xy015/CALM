import os
import glob
import time
import math
import random
import copy

import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F

class myDataSet(Dataset):
    def __init__(self, 
                 idx_list, 
                 if_augmentation=True, 
                 annotations_files=[], 
                 parent_dirs=[], 
                 target_size=(224, 224), 
                 scaler=None, 
                 label_columns=['TLC实测值', 'FVC实测值', 'FEV1实测值'],
                 aug_factor=1,  # 数据增强倍数
                 enable_dynamic_aug=True,  # 启用动态增强
                 seed=567,  # 随机种子
                 num_slices_per_view=15,  # 每个视图的切片数
                 cache_dir=None  # 缓存目录
                 ):
        super(myDataSet, self).__init__()
        self.idx_list = idx_list
        self.if_augmentation = if_augmentation
        self.target_size = target_size
        self.scaler = scaler
        self.label_columns = label_columns
        self.annotations_files = annotations_files
        self.parent_dirs = parent_dirs
        self.aug_factor = aug_factor
        self.enable_dynamic_aug = enable_dynamic_aug
        self.seed = seed
        self.views = ['axial', 'coronal', 'sagittal']
        self.num_slices_per_view = num_slices_per_view
        self.cache_dir = cache_dir
        
        # 增强参数配置
        self.aug_prob = 1.0
        self.rotation_range = (-15, 15)  # 角度范围
        self.shift_range = (-15, 15)    # 像素范围（需要归一化）
        self.zoom_range = (0.85, 1.15)
        
        max_shift = max(self.target_size)
        max_translate = abs(self.shift_range[1]) / max_shift  

        self.aug_transform = T.Compose([
            T.RandomAffine(
                degrees=self.rotation_range,
                translate=(max_translate, max_translate),  # 只传正值
                scale=self.zoom_range,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=0
             )
        ])
        self._validate_inputs()
        self.df_info = self._load_annotations()
        self.patient_list, self.missing_ids = self._build_patient_list()
        self.missing_npz_ids = [pid for pid, reason in self.missing_ids if reason == "无NPZ文件"]
        
        # 统计数据量
        self.original_length = len(self.patient_list)
        self.augmented_length = self.original_length * self.aug_factor
        print(f"原始有效数据量: {self.original_length}, 增强后数据量: {self.augmented_length}")
        print(f"增强模式: {'动态增强 (每次都不同)' if self.enable_dynamic_aug else '固定增强 (可重现)'}")
        
        # 仅在固定增强模式下初始化预设配置
        if not self.enable_dynamic_aug:
            self._init_fixed_augmentation_configs()
            
        # 创建缓存目录
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _init_fixed_augmentation_configs(self):
        """初始化每个原始样本的固定增强配置"""
        print("初始化固定增强配置...")
        self.augmentation_configs = {}
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        for original_idx in range(self.original_length):
            configs = []
            for aug_version in range(1, self.aug_factor):
                config = {
                    'angle': random.uniform(*self.rotation_range),
                    'translate': (random.uniform(*self.shift_range)/self.target_size[0], 
                                 random.uniform(*self.shift_range)/self.target_size[1]),
                    'scale': random.uniform(*self.zoom_range)
                }
                configs.append(config)
            self.augmentation_configs[original_idx] = configs
        print(f"已为 {self.original_length} 个样本生成固定增强配置")

    def _load_and_preprocess_images(self, pid, image_dir, aug_config=None):
        """向量化加载并预处理图像，支持应用指定的增强配置"""
        all_views = []
        
        # 向量化加载所有视图
        for direction in self.views:
            file_path = os.path.join(image_dir, f"{pid}_{direction}.npz")
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    slices = data['data']  # 形状: [num_slices, H, W]
                    
                    # 向量化调整所有切片尺寸
                    resized_slices = np.stack([
                        cv2.resize(s, self.target_size) for s in slices
                    ], axis=0)  # 形状: [num_slices, H, W]
                    
                    # 添加通道维度
                    if resized_slices.ndim == 3:  # 如果没有通道维度
                        resized_slices = resized_slices[..., np.newaxis]  # [num_slices, H, W, 1]
                        
                    all_views.append(resized_slices)
            except Exception as e:
                print(f"加载 {file_path} 失败: {e}")
                raise

        # 组合三个视图的对应切片为3通道图像
        # 形状: [num_slices, H, W, 3]
        combined_slices = np.concatenate(all_views, axis=-1)
        
        # 转为PyTorch张量 [num_slices, 3, H, W]
        tensor_slices = torch.from_numpy(combined_slices).permute(0, 3, 1, 2).float()
        
        # 应用增强
        if aug_config and aug_config != {}:
            # 应用PyTorch增强
            if self.enable_dynamic_aug:
                # 动态增强
                tensor_slices = self.aug_transform(tensor_slices)
            else:
                # 固定增强
                angle = aug_config['angle']
                translate = aug_config['translate']
                scale = aug_config['scale']
                
                # 创建仿射变换矩阵
                theta = torch.tensor([
                    [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), translate[0]],
                    [scale * np.sin(np.radians(angle)),  scale * np.cos(np.radians(angle)), translate[1]]
                ], dtype=torch.float32)
                
                # 扩展为批次变换矩阵 [num_slices, 2, 3]
                theta = theta.expand(tensor_slices.size(0), -1, -1)
                
                # 应用仿射变换
                grid = F.affine_grid(theta, tensor_slices.size(), align_corners=False)
                tensor_slices = F.grid_sample(tensor_slices, grid, align_corners=False)
        
        return tensor_slices

    def __len__(self):
        """返回增强后总数据量"""
        return self.augmented_length if self.if_augmentation else self.original_length

    def __getitem__(self, idx):
        """通过idx获取样本，支持动态或固定增强"""
        if self.if_augmentation:
            original_idx = idx // self.aug_factor
            aug_version = idx % self.aug_factor
        else:
            original_idx = idx
            aug_version = 0  # 非增强模式仅使用原始数据

        pid, ann_file_path, image_dir, matched_df = self.patient_list[original_idx]
        
        # 获取原始标签
        labels = self._load_and_standardize_labels(pid, matched_df)
        
        # 检查缓存
        if self.cache_dir and not self.if_augmentation:  # 仅对非增强数据缓存
            cache_path = os.path.join(self.cache_dir, f"{pid}_{aug_version}.pt")
            if os.path.exists(cache_path):
                cached_data = torch.load(cache_path)
                # 确保缓存数据格式正确
                if isinstance(cached_data, tuple) and len(cached_data) == 4:
                    return cached_data
        
        # 根据模式应用增强配置
        if aug_version == 0:  # 原始数据
            images = self._load_and_preprocess_images(pid, image_dir, None)
        else:  # 增强数据
            if self.enable_dynamic_aug:
                # 动态增强（使用PyTorch的随机增强）
                images = self._load_and_preprocess_images(pid, image_dir, {"dynamic": True})
            else:
                # 使用预生成的固定配置
                aug_config = self.augmentation_configs[original_idx][aug_version - 1]
                images = self._load_and_preprocess_images(pid, image_dir, aug_config)

        # 保存到缓存
        if self.cache_dir and not self.if_augmentation:
            torch.save((images, labels, pid, aug_version), cache_path)
            
        return images, labels, pid, aug_version

    def _validate_inputs(self):
        """校验输入参数合法性"""
        # 校验标注文件与图像目录数量一致
        if len(self.annotations_files) != len(self.parent_dirs):
            raise ValueError("annotations_files与parent_dirs数量必须一致")
        
        # 校验标签列数量（必须为3列）
        if len(self.label_columns) != 3:
            raise ValueError(f"label_columns必须包含3个标签列，当前传入{len(self.label_columns)}个")

    def _load_annotations(self):
        """加载标注文件并记录文件路径"""
        df_info = []
        for ann_file in self.annotations_files:
            df = pd.read_csv(ann_file, dtype={'new_ID': str})  # 保留ID前导零
            # 检查当前文件是否包含所有标签列
            missing_cols = [col for col in self.label_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"标注文件 {ann_file} 缺少标签列: {missing_cols}")
            df_info.append( (df, ann_file) )  # 存储 (DataFrame, 标注文件路径)
        return df_info

    def _build_patient_list(self):
        """构建有效患者列表（存储完整上下文信息）"""
        patient_list = []  # 存储 (pid, ann_file_path, image_dir, matched_df)
        missing_ids = []   # 存储 (pid, 无效原因)

        for pid in self.idx_list:
            matched_ann_file = None  # 匹配的标注文件路径（字符串）
            matched_df = None        # 匹配的原始DataFrame
            matched_image_dir = None # 匹配的图像目录（字符串）

            # 遍历所有标注文件（按优先级顺序查找）
            for df, ann_file in self.df_info:
                if pid in df['new_ID'].values:  # 检查ID是否存在于原始DataFrame中
                    matched_ann_file = ann_file  # 记录标注文件路径（关键修正）
                    matched_df = df              # 记录原始DataFrame
                    # 通过标注文件路径找到对应的图像目录
                    dir_index = self.annotations_files.index(ann_file)
                    matched_image_dir = self.parent_dirs[dir_index]
                    break  # 找到后停止，确保优先级

            # 情况1：无标签数据（未在任何标注文件中找到ID）
            if not matched_ann_file:
                missing_ids.append( (pid, "无标签数据") )
                continue

            # 情况2：检查是否存在NPZ文件
            if not self._check_image_exists(pid, matched_image_dir):
                missing_ids.append( (pid, "无NPZ文件") )
                continue

            # 有效患者：存储完整上下文信息
            patient_list.append( (pid, matched_ann_file, matched_image_dir, matched_df) )

        return patient_list, missing_ids

    def _check_image_exists(self, patient_id, image_dir):
        """检查患者的NPZ文件是否存在"""
        if not os.path.exists(image_dir):
            return False
    
        all_files_exist = True
        for view in self.views:
            file_path = os.path.join(image_dir, f"{patient_id}_{view}.npz")
            if not os.path.exists(file_path):
                all_files_exist = False
                break
        return all_files_exist

    def _load_and_standardize_labels(self, pid, df):
        """加载并标准化标签数据"""
        # 提取原始标签
        label_row = df[df['new_ID'] == pid][self.label_columns]
        if label_row.empty:
            raise ValueError(f"患者ID {pid} 在标注文件中无标签数据")
        raw_label = label_row.values.astype(np.float32).squeeze()  # 形状：(3,)

        # 检查标签维度
        if raw_label.shape[0] != 3:
            raise ValueError(f"患者ID {pid} 标签维度错误，应为3，实际为 {raw_label.shape[0]}")

        # 应用标准化
        if self.scaler is not None:
            # scaler需要2D输入（n_samples=1, n_features=3）
            standardized_label = self.scaler.transform(raw_label.reshape(1, -1)).squeeze()
            return torch.from_numpy(standardized_label).float()
        else:
            return torch.from_numpy(raw_label).float()

    def get_missing_ids(self):
        """返回所有无效ID列表（格式：[(pid, 无效原因), ...]）"""
        return self.missing_ids
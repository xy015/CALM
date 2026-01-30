import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from lungmask import LMInferer
from skimage.morphology import binary_closing, binary_erosion, binary_dilation, disk, ball

inferer = LMInferer(
        modelname="R231",
        modelpath="./unet_r231-d5d2fc3d.pth"
    )
print("✅ R231 模型已加载")
def load_dicom_series(dicom_dir):
    """从DICOM文件夹加载3D图像"""
    series_reader = sitk.ImageSeriesReader()
    dicom_files = series_reader.GetGDCMSeriesFileNames(dicom_dir)
    series_reader.SetFileNames(dicom_files)
    return series_reader.Execute()

#将3D CT图像重采样（1mm*1mm*1mm）
def resample_to_isotropic(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)
#根据肺分割mask裁剪CT图像的ROI区域
def crop_lung_region(ct_arr, mask_np, margin=5):
    
    coords = np.array(np.where(mask_np > 0))
    if coords.size == 0:
        z, y, x = ct_arr.shape
        return ct_arr, mask_np, np.array([0,0,0]), np.array([z-1, y-1, x-1])

    min_coords = coords.min(axis=1)
    max_coords = coords.max(axis=1)

    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, np.array(mask_np.shape) - 1)

    ct_cropped = ct_arr[min_coords[0]:max_coords[0]+1,
                        min_coords[1]:max_coords[1]+1,
                        min_coords[2]:max_coords[2]+1]
    mask_cropped = mask_np[min_coords[0]:max_coords[0]+1,
                          min_coords[1]:max_coords[1]+1,
                          min_coords[2]:max_coords[2]+1]

    return ct_cropped, mask_cropped, min_coords, max_coords

def fix_image_orientation(image):
    
    direction = np.array(image.GetDirection()).reshape((3, 3))
    
    if direction[2, 2] < 0:
        arr = sitk.GetArrayFromImage(image)
        arr = arr[::-1, :, :]
        direction[2, :] *= -1
        new_direction = tuple(direction.flatten())
        new_image = sitk.GetImageFromArray(arr)
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(new_direction)
        new_image.SetOrigin(image.GetOrigin())
        return new_image
    else:
        return image

def postprocess_lung_mask(mask_np, kernel_size=3):
    """
    肺分割结果的形态学后处理
    :param mask_np: 原始分割mask（二值化，0=背景，1=肺）
    :param kernel_size: 形态学操作的核大小（推荐3-5）
    :return: 后处理后的mask
    """
    # 1. 二值化（确保是0-1矩阵）
    binary_mask = (mask_np > 0).astype(np.uint8)
    
    # 2. 根据输入维度选择合适的核
    if binary_mask.ndim == 2:
        # 二维数据使用disk核
        kernel = disk(kernel_size)
    else:
        # 三维数据使用ball核
        kernel = ball(kernel_size)
    
    # 3. 闭运算：填充小空洞，连接邻近区域
    closed = binary_closing(binary_mask, kernel)
    
    # 4. 开运算：消除小噪声，平滑边缘
    opened = binary_erosion(closed, kernel)  # 先腐蚀
    opened = binary_dilation(opened, kernel)  # 再膨胀
    
    # 5. 去除小面积区域（如误分割的小斑点）
    from skimage.measure import label, regionprops
    labeled = label(opened)
    for region in regionprops(labeled):
        if region.area < 100:  # 面积阈值
            for coord in region.coords:
                labeled[coord[0], coord[1]] = 0
    opened = (labeled > 0).astype(np.uint8)
    
    return opened

def segment_ct_r231(input_dicom_dir, output_dir, margin=5, min_slices=10, target_slices=15):
    """
    对CT图像进行肺部分割并提取切片
    
    参数:
    input_nii: 输入CT文件路径
    output_dir: 输出目录
    margin: 裁剪肺部区域时的边距
    min_slices: 每个方向至少需要的切片数
    target_slices: 每个方向目标提取的切片数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建记录文件
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, "slice_stats.txt")
    try:
        ct = load_dicom_series(input_dicom_dir)  # 加载DICOM序列
        print(f"✅ 输入 DICOM 文件夹: {input_dicom_dir}")
    except Exception as e:
        print(f"❌ 读取DICOM失败: {str(e)}")
        with open(stats_file, "a") as f:
            f.write(f"❌ 读取DICOM失败: {str(e)}\n")
        return None, None    
    with open(stats_file, "a") as f:
        f.write(f"Processing: {input_dicom_dir}\n")

    # 重采样到 1mm x 1mm x 1mm
    ct = resample_to_isotropic(ct, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
    ct_arr = sitk.GetArrayFromImage(ct)  # 原始HU值数组，未归一化
    print("✅ 已完成 1mm x 1mm x 1mm 重采样，新的 shape:", ct_arr.shape)
   
    # 执行分割（需要HU值，因此传原始ct）
    mask_np = inferer.apply(ct)
    print("✅ Mask (numpy) 形状:", mask_np.shape)
    binary_mask = (mask_np > 0).astype(np.uint8)  
    binary_mask = np.isin(mask_np, [1, 2]).astype(np.uint8)  

    # 形态学后处理
    processed_mask = postprocess_lung_mask(binary_mask, kernel_size=3)

    # 替换原始mask（后续流程用 processed_mask 代替 mask_np）
    mask_np = processed_mask
    
    # 保存Mask
    mask_img = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    mask_img.SetSpacing(ct.GetSpacing())
    mask_img.SetDirection(ct.GetDirection())
    mask_img.SetOrigin(ct.GetOrigin())
    patient_id = os.path.basename(input_dicom_dir)  # 从文件夹名获取ID

    # 提取肺部区域（使用原始CT值，非归一化）
    lung_only_arr = ct_arr * (mask_np > 0) 
    
    # 剪裁肺部区域（使用原始CT值）
    ct_cropped_arr, mask_cropped_np, min_coord, max_coord = crop_lung_region(ct_arr, mask_np, margin=margin)
    print(f"✅ 肺部区域剪裁后的shape: {ct_cropped_arr.shape}")
    
    # ------------------- 新增切片提取与标准化逻辑 -------------------
    # 定义三个视图方向（轴向Z、冠状Y、矢状X）
    views = {
        "axial": {"axis": 0, "name": "axial"},    # 轴向：Z轴切片
        "coronal": {"axis": 1, "name": "coronal"},  # 冠状面：Y轴切片
        "sagittal": {"axis": 2, "name": "sagittal"} # 矢状面：X轴切片
    }

    # 计算各方向中心索引
    center_indices = {
        "axial": ct_cropped_arr.shape[0] // 2,    # Z轴中心（轴向切片方向）
        "coronal": ct_cropped_arr.shape[1] // 2,  # Y轴中心（冠状面切片方向）
        "sagittal": ct_cropped_arr.shape[2] // 2  # X轴中心（矢状面切片方向）
    }
    
    # 获取实际的体素间距
    spacing = ct.GetSpacing()  # 使用重采样后的图像间距
    print(f"体素间距: {spacing}")
    
    # 计算各方向中心对应的物理坐标（单位：mm）
    axial_center = center_indices["axial"] * spacing[2]  # Z轴间距
    coronal_center = center_indices["coronal"] * spacing[1]  # Y轴间距
    sagittal_center = center_indices["sagittal"] * spacing[0]  # X轴间距
    
    # 验证是否一致（误差在容差内）
    tolerance = 1e-6
    if (abs(axial_center - coronal_center) < tolerance and
        abs(axial_center - sagittal_center) < tolerance):
        print("✅ 三个视图的中心在物理空间中一致")
    else:
        print(f"⚠️ 三个视图的中心存在偏移:")
        print(f"  - 轴向中心: {axial_center:.2f} mm")
        print(f"  - 冠状面中心: {coronal_center:.2f} mm")
        print(f"  - 矢状面中心: {sagittal_center:.2f} mm")
    
    view_results = {}
    
    # 记录每个方向的切片数量
    slice_counts = {}
    
    # 检查是否有任何方向的切片数量少于最小值
    insufficient_slices = False
    insufficient_views = []

    # 全局标准化：计算整个样本肺部区域的均值和方差
    all_lung_pixels = ct_cropped_arr[mask_cropped_np > 0]
    if all_lung_pixels.size == 0:
        raise ValueError("No lung region found in the cropped data.")
    global_mean = np.mean(all_lung_pixels)
    global_std = np.std(all_lung_pixels)

    # 计算全局Z-score的最小值和最大值
    z_scores = (all_lung_pixels - global_mean) / global_std
    z_min = np.min(z_scores)
    z_max = np.max(z_scores)

    # 防止除零错误（当所有像素值相同时）
    if z_min == z_max:
        z_min = -1.0 
        z_max = 1.0
    
    # 检查每个方向的切片数量
    for view in views.values():
        axis = view["axis"]
        view_name = view["name"]
        center_idx = center_indices[view_name]
        
        # 计算该方向上可用的切片数量
        available_slices = ct_cropped_arr.shape[axis]
        slice_counts[view_name] = available_slices
        
        print(f"📊 {view_name} 方向可用切片数量: {available_slices}/{target_slices}")
        
        # 记录切片数量
        with open(stats_file, "a") as f:
            f.write(f"{view_name}方向切片数量: {available_slices}\n")
        
        # 检查是否少于最小要求
        if available_slices < min_slices:
            insufficient_slices = True
            insufficient_views.append(view_name)
            print(f"❌ {view_name} 方向切片数量 {available_slices} 少于最小要求 {min_slices}")
        elif available_slices < target_slices:
            print(f"⚠️ {view_name} 方向切片数量 {available_slices} 少于目标数量 {target_slices}")
    
    # 如果有任何方向的切片数量少于最小值，记录并退出
    if insufficient_slices:
        print(f"\n❌ 样本 {patient_id} 不符合要求：以下方向切片数量少于 {min_slices}: {', '.join(insufficient_views)}")
        with open(stats_file, "a") as f:
            f.write(f"❌ 样本不符合要求：以下方向切片数量少于 {min_slices}: {', '.join(insufficient_views)}\n\n")
        
        # 记录到低质量样本列表
        low_quality_file = os.path.join(stats_dir, "low_quality_samples.txt")
        with open(low_quality_file, "a") as f:
            f.write(f"{patient_id}: 切片不足 ({', '.join([f'{v}:{slice_counts[v]}' for v in insufficient_views])})\n")
        
        # 返回None表示样本不合格
        return None, None
    
    # 记录切片数量在10-15之间的样本
    if any(min_slices <= count < target_slices for count in slice_counts.values()):
        suboptimal_file = os.path.join(stats_dir, "suboptimal_samples.txt")
        with open(suboptimal_file, "a") as f:
            slice_info = ", ".join([f"{v}:{slice_counts[v]}" for v in views.keys()])
            f.write(f"{patient_id}: {slice_info}\n")
    
    # 为每个方向提取切片
    for view in views.values():
        axis = view["axis"]
        view_name = view["name"]
        center_idx = center_indices[view_name]
        
        # 计算该方向上可用的切片数量
        available_slices = ct_cropped_arr.shape[axis]
        
        # 生成切片索引：以中心为基准，尽量提取target_slices张切片
        if available_slices >= target_slices:
            
            total_range = (target_slices - 1) * 10
            start_idx = max(0, center_idx - total_range // 2)
            end_idx = min(available_slices, start_idx + total_range)
            
           
            if end_idx - start_idx > 0:
                step = max(1, (end_idx - start_idx) // (target_slices - 1))
                slice_indices = np.arange(start_idx, end_idx + 1, step)
                
                if len(slice_indices) > target_slices:
                    slice_indices = slice_indices[:target_slices]
                elif len(slice_indices) < target_slices:
                    while len(slice_indices) < target_slices:
                        if start_idx > 0:
                            start_idx -= 1
                            slice_indices = np.insert(slice_indices, 0, start_idx)
                        elif end_idx < available_slices - 1:
                            end_idx += 1
                            slice_indices = np.append(slice_indices, end_idx)
                        else:
                            break
            else:
                # 范围太小，直接取中心的target_slices个切片
                slice_indices = np.arange(max(0, center_idx - target_slices//2), 
                                         min(available_slices, center_idx + target_slices//2 + 1))
        else:
            # 切片不足，使用所有可用切片
            slice_indices = np.arange(available_slices)
        
        slice_indices = np.clip(slice_indices, 0, available_slices - 1)
        
        # 提取该方向的所有切片和对应的Mask
        slices = []
        mask_slices = []
        for idx in slice_indices:
            if axis == 0:  # 轴向切片，形状(y, x)
                slice_arr = ct_cropped_arr[int(idx), :, :]
                slice_mask = mask_cropped_np[int(idx), :, :]
            elif axis == 1:  # 冠状面切片，形状(z, x)
                slice_arr = ct_cropped_arr[:, int(idx), :]
                slice_mask = mask_cropped_np[:, int(idx), :]
                slice_arr = np.rot90(slice_arr, k=2)  
                slice_mask = np.rot90(slice_mask, k=2)
            else:  # 矢状面切片，形状(z, y)
                slice_arr = ct_cropped_arr[:, :, int(idx)]
                slice_mask = mask_cropped_np[:, :, int(idx)]
                slice_arr = np.rot90(slice_arr, k=2)  
                slice_mask = np.rot90(slice_mask, k=2)
                slice_arr = np.fliplr(slice_arr)  
                slice_mask = np.fliplr(slice_mask)
            
            # 提取肺部区域并标准化
            lung_slice = slice_arr * (slice_mask > 0)
            lung_slice_standardized = (lung_slice - global_mean) / global_std  
            # 基于全局Z-score极值的Min-Max归一化
            normalized = (lung_slice_standardized - z_min) / (z_max - z_min)
            normalized = np.clip(normalized, 0, 1).astype(np.float32)   
            
            slices.append(normalized)
            mask_slices.append(slice_mask)  # 保存Mask切片用于可视化和占比计算
        
        # 保存为npz文件（包含数据和Mask）
        view_results[view_name] = {
            "data": np.stack(slices, axis=0),
            "mask": np.stack(mask_slices, axis=0),
            "indices": np.array(slice_indices)  
        }
        npz_path = os.path.join(output_dir, f"{patient_id}_{view_name}.npz")
        np.savez(npz_path, **view_results[view_name])
        print(f"✅ 已保存 {view_name} 视图 {len(slices)} 张切片到 {npz_path}")

    # 可视化
    def visualize_slices(output_dir, patient_id, view_name, num_display=min(target_slices, 5)):
        """可视化指定视图的切片并计算肺部占比"""
        npz_path = os.path.join(output_dir, f"{patient_id}_{view_name}.npz")
        data = np.load(npz_path)
        slices = data["data"]
        masks = data["mask"]
        slice_indices = data["indices"]  # 加载原始索引
        num_slices = slices.shape[0]
        
        # 计算每张切片的肺部占比
        lung_ratios = [np.sum(mask > 0) / mask.size for mask in masks]
        avg_ratio = np.mean(lung_ratios)
        
        # 计算数据统计量（所有切片合并统计）
        all_pixels = slices.flatten()
        min_val = np.min(all_pixels)
        max_val = np.max(all_pixels)
        mean_val = np.mean(all_pixels)
        std_val = np.std(all_pixels)

        print(f"\n--- {view_name} 视图分析 ---")
        print(f"总切片数: {num_slices}")
        print(f"平均肺部占比: {avg_ratio*100:.2f}%")
        print("前5张切片占比:", ["{:.2f}%".format(ratio*100) for ratio in lung_ratios[:5]])
    
        print("\n--- 数据统计 ---")
        print(f"所有切片合并后:")
        print(f"最小值: {min_val:.4f}")
        print(f"最大值: {max_val:.4f}")
        print(f"均值: {mean_val:.4f}")
        print(f"标准差: {std_val:.4f}")

        plt.figure(figsize=(15, 4 * num_display))
        for i in range(min(num_display, num_slices)):
            original_idx = slice_indices[i] 
            ax = plt.subplot(num_display, 2, 2*i+1)
            # 显示标准化后的肺部区域
            plt.imshow(slices[i], cmap='gray', vmin=0, vmax=1)
            plt.title(f"Slice {i+1}_{original_idx}(normalized)")
            plt.axis('off')
            
            ax = plt.subplot(num_display, 2, 2*i+2)
            # 显示原始CT切片
            plt.imshow(slices[i], cmap='gray', vmin=slices[i].min(), vmax=slices[i].max())
            # 叠加红色Mask
            plt.imshow(masks[i], cmap='jet', alpha=0.3)
            plt.title(f"Slice {i+1} (ratio: {lung_ratios[i]*100:.2f}%)")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    with open(stats_file, "a") as f:
        f.write(f"✅ 样本合格: {patient_id}\n")
        f.write(f"切片数量统计: {', '.join([f'{v}:{slice_counts[v]}' for v in views.keys()])}\n\n")

    return ct, mask_img  

def process_samples_from_csv(csv_file, ct_path, output_dir, id_column="new_ID", 
                             min_slices=10, target_slices=15, skip_existing=True):
    """
    从CSV文件读取样本ID列表，批量处理CT样本
    
    参数:
    csv_file: CSV文件路径，包含样本ID列表
    ct_path: CT文件所在目录
    output_dir: 输出目录
    id_column: CSV文件中包含样本ID的列名
    min_slices: 每个方向至少需要的切片数
    target_slices: 每个方向目标提取的切片数
    skip_existing: 是否跳过已处理的样本
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    # 初始化统计文件
    with open(os.path.join(stats_dir, "slice_stats.txt"), "w") as f:
        f.write("=== 切片数量统计 ===\n\n")
    
    with open(os.path.join(stats_dir, "low_quality_samples.txt"), "w") as f:
        f.write("=== 低质量样本（切片数量少于10）===\n\n")
    
    with open(os.path.join(stats_dir, "suboptimal_samples.txt"), "w") as f:
        f.write("=== 次优样本（切片数量在10-15之间）===\n\n")
    
    print(f"读取CSV文件: {csv_file}")
    df = pd.read_csv(csv_file,dtype={'new_ID': str})

    if id_column not in df.columns:
        raise ValueError(f"CSV文件中未找到列 '{id_column}'")
    
    # 获取样本ID列表
    sample_ids = df[id_column].tolist()
    print(f"找到 {len(sample_ids)} 个样本ID")

    total_samples = len(sample_ids)
    processed_samples = 0
    failed_samples = 0
    qualified_samples = 0
    
    for sample_id in tqdm(sample_ids, desc="处理样本"):
        try:
            print(f"\n--- 处理样本: {sample_id} ---")
            
            # 构建DICOM文件夹路径 
            dicom_dir = os.path.join(ct_path, str(sample_id)) 
            if not os.path.isdir(dicom_dir):
                print(f"❌ 未找到DICOM文件夹: {dicom_dir}")
                with open(os.path.join(stats_dir, "missing_files.txt"), "a") as f:
                    f.write(f"{sample_id}\n")
                failed_samples += 1
                continue
            
            # 检查是否已处理
            if skip_existing:
                output_exists = False
                for view in ["axial", "coronal", "sagittal"]:
                    if os.path.exists(os.path.join(output_dir, f"{sample_id}_{view}.npz")):
                        output_exists = True
                        break
                if output_exists:
                    print(f"⏭️ 样本 {sample_id} 已处理，跳过")
                    processed_samples += 1
                    continue
            
            
            # 处理样本
            ct, mask_img = segment_ct_r231(
                dicom_dir,  # 传入文件夹路径而非单个文件
                output_dir, 
                margin=5, 
                min_slices=min_slices, 
                target_slices=target_slices
            )
            
            # 统计处理结果
            processed_samples += 1
            if ct is not None and mask_img is not None:
                qualified_samples += 1
                print(f"✅ 样本 {sample_id} 处理完成且符合要求")
            else:
                print(f"⚠️ 样本 {sample_id} 处理完成但不符合要求")
        
        except Exception as e:
            print(f"❌ 处理样本 {sample_id} 时出错: {str(e)}")
            with open(os.path.join(stats_dir, "error_log.txt"), "a") as f:
                f.write(f"{sample_id}: {str(e)}\n")
            failed_samples += 1
    
    # 输出处理统计
    print("\n--- 处理统计 ---")
    print(f"总样本数: {total_samples}")
    print(f"成功处理: {processed_samples}")
    print(f"符合要求: {qualified_samples}")
    print(f"未找到CT文件: {total_samples - processed_samples}")
    print(f"处理失败: {failed_samples}")
    
    # 写入统计摘要
    with open(os.path.join(stats_dir, "summary.txt"), "w") as f:
        f.write("=== 批处理统计摘要 ===\n\n")
        f.write(f"总样本数: {total_samples}\n")
        f.write(f"成功处理: {processed_samples}\n")
        f.write(f"符合要求: {qualified_samples}\n")
        f.write(f"未找到CT文件: {total_samples - processed_samples}\n")
        f.write(f"处理失败: {failed_samples}\n")

if __name__ == "__main__":
    csv_file = "./modified_data_975.csv"  # 包含样本ID的CSV文件
    ct_path = "./RealData/CT/"  # DICOM根目录，每个样本一个子文件夹
    output_dir = "./CT_Images/ROI_975/"  # 输出目录
    id_column = "new_ID"  # CSV文件中包含样本ID的列名
    
    process_samples_from_csv(
        csv_file=csv_file,
        ct_path=ct_path,
        output_dir=output_dir,
        id_column=id_column,
        min_slices=10,
        target_slices=15,
        skip_existing=True
    )

"""
MRI数据预处理脚本
"""
import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def resample_image(
    image: sitk.Image,
    new_spacing: Tuple[float, float, float],
    interpolator=sitk.sitkLinear
) -> sitk.Image:
    """重采样图像到新的spacing"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    new_size = [
        int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / new_spacing[2]))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)


def normalize_intensity(image_array: np.ndarray, method: str = "zscore") -> np.ndarray:
    """强度归一化"""
    if method == "zscore":
        mean = np.mean(image_array[image_array > 0])
        std = np.std(image_array[image_array > 0])
        if std > 0:
            image_array = (image_array - mean) / std
    elif method == "minmax":
        min_val = np.min(image_array[image_array > 0])
        max_val = np.max(image_array[image_array > 0])
        if max_val > min_val:
            image_array = (image_array - min_val) / (max_val - min_val)
    
    return image_array


def preprocess_mri(
    file_path: str,
    target_shape: Tuple[int, int, int] = (182, 218, 182),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity_norm: str = "zscore",
    skull_strip: bool = False
) -> np.ndarray:
    """
    预处理MRI图像
    
    Args:
        file_path: MRI文件路径
        target_shape: 目标形状
        target_spacing: 目标spacing
        intensity_norm: 强度归一化方法
        skull_strip: 是否进行颅骨剥离（简化版本，使用阈值）
        
    Returns:
        预处理后的3D数组
    """
    # 读取图像
    image = sitk.ReadImage(file_path)
    
    # 重采样
    image = resample_image(image, target_spacing)
    
    # 转换为numpy数组
    image_array = sitk.GetArrayFromImage(image)
    
    # 简单的颅骨剥离（使用阈值）
    if skull_strip:
        threshold = np.percentile(image_array[image_array > 0], 5)
        image_array[image_array < threshold] = 0
    
    # 强度归一化
    image_array = normalize_intensity(image_array, method=intensity_norm)
    
    # 裁剪或填充到目标形状
    current_shape = image_array.shape
    result = np.zeros(target_shape, dtype=np.float32)
    
    # 计算裁剪/填充范围
    for i in range(3):
        if current_shape[i] > target_shape[i]:
            start = (current_shape[i] - target_shape[i]) // 2
            end = start + target_shape[i]
            if i == 0:
                image_array = image_array[start:end, :, :]
            elif i == 1:
                image_array = image_array[:, start:end, :]
            else:
                image_array = image_array[:, :, start:end]
        elif current_shape[i] < target_shape[i]:
            pad_start = (target_shape[i] - current_shape[i]) // 2
            pad_end = target_shape[i] - current_shape[i] - pad_start
            if i == 0:
                image_array = np.pad(image_array, ((pad_start, pad_end), (0, 0), (0, 0)), mode='constant')
            elif i == 1:
                image_array = np.pad(image_array, ((0, 0), (pad_start, pad_end), (0, 0)), mode='constant')
            else:
                image_array = np.pad(image_array, ((0, 0), (0, 0), (pad_start, pad_end)), mode='constant')
    
    result = image_array[:target_shape[0], :target_shape[1], :target_shape[2]]
    
    return result


class MRIDataset(Dataset):
    """MRI数据集类"""
    
    def __init__(
        self,
        data_paths: list,
        labels: list,
        slice_axis: int = 1,  # 0=sagittal, 1=coronal, 2=axial
        augment: bool = False,
        random_flip: bool = True,
        random_rotation: float = 15.0,
        random_scale: Tuple[float, float] = (0.9, 1.1),
        random_crop: bool = True
    ):
        self.data_paths = data_paths
        self.labels = labels
        self.slice_axis = slice_axis
        self.augment = augment
        
        # 数据增强
        if augment:
            transform_list = []
            if random_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            if random_rotation > 0:
                transform_list.append(transforms.RandomRotation(random_rotation))
            if random_scale:
                transform_list.append(transforms.RandomAffine(0, scale=random_scale))
            transform_list.append(transforms.ToTensor())
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # 预处理
        volume = preprocess_mri(file_path)
        
        # 提取中间切片
        if self.slice_axis == 0:
            slice_idx = volume.shape[0] // 2
            slice_img = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_idx = volume.shape[1] // 2
            slice_img = volume[:, slice_idx, :]
        else:
            slice_idx = volume.shape[2] // 2
            slice_img = volume[:, :, slice_idx]
        
        # 归一化到[0, 1]
        slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8)
        slice_img = (slice_img * 255).astype(np.uint8)
        
        # 转换为PIL Image并转换为RGB（3通道）
        from PIL import Image
        pil_img = Image.fromarray(slice_img, mode='L')
        pil_img = pil_img.convert('RGB')
        
        # 数据增强
        slice_tensor = self.transform(pil_img)
        
        return slice_tensor, label

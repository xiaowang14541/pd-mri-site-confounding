"""
数据收集脚本：从原始数据目录收集所有MRI文件并分配标签
"""
import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd


def collect_data_paths(raw_data_dir: str) -> List[Tuple[str, int, str]]:
    """
    收集所有MRI文件路径、标签和数据集来源
    
    Args:
        raw_data_dir: 原始数据目录
        
    Returns:
        List of (file_path, label, dataset) tuples
        label: 0 = HC (健康对照), 1 = PD (帕金森病)
        dataset: 数据集来源标识
    """
    data_paths = []
    
    # 查找所有nii.gz文件
    pattern = os.path.join(raw_data_dir, "**", "*.nii.gz")
    all_files = glob.glob(pattern, recursive=True)
    
    print(f"找到 {len(all_files)} 个MRI文件")
    
    # 根据文件路径判断标签和数据集
    for file_path in all_files:
        # 获取完整路径的各个部分
        path_parts = file_path.split(os.sep)
        
        # 判断数据集来源
        dataset = None
        if 'ds000245' in file_path:
            dataset = 'ds000245'
        elif 'ODP' in file_path or 'ODN' in file_path or 'CTL' in file_path:
            # ds000245的数据直接在根目录下，通过subject ID识别
            dataset = 'ds000245'
        elif 'ds004392' in file_path or 'PD_Cognition' in file_path:
            dataset = 'ds004392'
        elif 'ds004471' in file_path or 'LHSCPD' in file_path:
            dataset = 'ds004471'
        elif 'ds003592' in file_path or 'HealthyControls' in file_path:
            dataset = 'ds003592'
        
        # 判断是PD还是HC
        label = None
        
        # 方法1: 检查subject ID前缀（ds000245）
        subject_id = None
        for part in path_parts:
            if part.startswith('sub-'):
                subject_id = part
                break
        
        if subject_id:
            # ds000245: sub-ODP* = PD, sub-ODN* 和 sub-CTL* = HC
            if 'ODP' in subject_id:
                label = 1  # PD
            elif 'ODN' in subject_id or 'CTL' in subject_id:
                label = 0  # HC
        
        # 方法2: 检查数据集目录名称
        if label is None:
            if dataset == 'ds004392':
                label = 1  # PD
            elif dataset == 'ds004471':
                label = 1  # PD
            elif dataset == 'ds003592':
                label = 0  # HC
        
        if label is not None and dataset is not None:
            data_paths.append((file_path, label, dataset))
    
    # 统计
    pd_count = sum(1 for _, label, _ in data_paths if label == 1)
    hc_count = sum(1 for _, label, _ in data_paths if label == 0)
    
    print(f"\n数据统计:")
    print(f"  PD患者: {pd_count} 人")
    print(f"  健康对照: {hc_count} 人")
    print(f"  总计: {len(data_paths)} 人")
    
    # 按数据集统计
    from collections import defaultdict
    dataset_stats = defaultdict(lambda: {'PD': 0, 'HC': 0})
    for _, label, dataset in data_paths:
        if label == 1:
            dataset_stats[dataset]['PD'] += 1
        else:
            dataset_stats[dataset]['HC'] += 1
    
    print(f"\n各数据集分布:")
    for ds, stats in sorted(dataset_stats.items()):
        print(f"  {ds}: PD={stats['PD']}, HC={stats['HC']}, Total={stats['PD']+stats['HC']}")
    
    return data_paths


def save_data_info(data_paths: List[Tuple[str, int, str]], output_path: str):
    """保存数据信息到CSV文件"""
    df = pd.DataFrame(data_paths, columns=['file_path', 'label', 'dataset'])
    df['subject_id'] = df['file_path'].apply(
        lambda x: os.path.basename(os.path.dirname(os.path.dirname(x)))
    )
    df['class'] = df['label'].apply(lambda x: 'PD' if x == 1 else 'HC')
    df = df[['subject_id', 'file_path', 'label', 'class', 'dataset']]
    df.to_csv(output_path, index=False)
    print(f"\n数据信息已保存到: {output_path}")
    return df


if __name__ == "__main__":
    raw_data_dir = "/master/wang_zi_wei/cursor/parkinson_mri_data"
    output_path = "/master/wang_zi_wei/cursor/pd_classification/data/data_info.csv"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data_paths = collect_data_paths(raw_data_dir)
    df = save_data_info(data_paths, output_path)
    
    print("\n前5条数据:")
    print(df.head())

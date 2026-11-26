"""
分类模型定义
"""
import torch
import torch.nn as nn
import torchvision.models as models


def get_model(model_name: str = "densenet121", num_classes: int = 2, pretrained: bool = True, dropout: float = 0.5):
    """
    获取预训练模型
    
    Args:
        model_name: 模型名称
        num_classes: 分类数量
        pretrained: 是否使用预训练权重
        dropout: dropout率
        
    Returns:
        模型
    """
    if model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == "efficientnet_b0":
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
        except ImportError:
            raise ImportError("需要安装timm: pip install timm")
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model

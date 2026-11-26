"""
训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import json
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=0.0001,
        weight_decay=0.01,
        num_epochs=50,
        early_stopping_patience=10,
        use_amp=False,
        save_dir=None,
        class_weights=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp
        self.save_dir = save_dir
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 损失函数（可选类别加权，缓解小样本类别不平衡）
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }
        
        # 早停
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(self.train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        return train_loss, train_acc
    
    def validate(self):
        """验证"""
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                val_preds.extend(preds)
                val_probs.extend(probs[:, 1])  # PD类的概率
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # 计算AUC
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except:
            val_auc = 0.0
        
        return val_loss, val_acc, val_auc, val_preds, val_labels, val_probs
    
    def train(self):
        """完整训练流程"""
        print(f"\n开始训练，共 {self.num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"使用AMP: {self.use_amp}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_auc, val_preds, val_labels, val_probs = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_auc)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            # 保存最佳模型
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                
                if self.save_dir:
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(self.best_model_state, os.path.join(self.save_dir, 'best_model.pth'))
                    print(f"✓ 保存最佳模型 (AUC: {val_auc:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n早停触发 (patience: {self.early_stopping_patience})")
                break
        
        # 加载最佳模型
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # AUC
        axes[2].plot(epochs, self.history['val_auc'], 'g-', label='Val AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Validation AUC')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存: {save_path}")
        
        plt.close()
    
    def evaluate(self):
        """最终评估"""
        val_loss, val_acc, val_auc, val_preds, val_labels, val_probs = self.validate()
        
        # 计算其他指标
        val_f1 = f1_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds)
        
        # 混淆矩阵
        cm = confusion_matrix(val_labels, val_preds)
        
        metrics = {
            'loss': float(val_loss),
            'accuracy': float(val_acc),
            'auc': float(val_auc),
            'f1_score': float(val_f1),
            'precision': float(val_precision),
            'recall': float(val_recall),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics

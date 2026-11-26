"""
Trainer for the domain-adversarial (DANN) DenseNet121, used for LODO validation
with a site-invariance objective. See src/models/dann_classifier.py for rationale.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm


class DANNTrainer:
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
        domain_loss_weight=1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = use_amp
        self.save_dir = save_dir
        self.domain_loss_weight = domain_loss_weight

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if use_amp else None

        self.total_iters = max(1, len(train_loader) * num_epochs)
        self.iter_count = 0

        self.history = {'train_loss': [], 'train_acc': [], 'train_domain_acc': [],
                         'val_loss': [], 'val_acc': [], 'val_auc': []}

        self.best_val_auc = -1.0
        self.patience_counter = 0
        self.best_model_state = None

    def _grl_lambda(self):
        p = self.iter_count / self.total_iters
        return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        domain_preds, domain_labels_all = [], []

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels, domains in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)

            lambda_ = self._grl_lambda()
            self.model.set_grl_lambda(lambda_)
            self.iter_count += 1

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    class_logits, domain_logits = self.model(images)
                    class_loss = self.class_criterion(class_logits, labels)
                    domain_loss = self.domain_criterion(domain_logits, domains)
                    loss = class_loss + self.domain_loss_weight * domain_loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                class_logits, domain_logits = self.model(images)
                class_loss = self.class_criterion(class_logits, labels)
                domain_loss = self.domain_criterion(domain_logits, domains)
                loss = class_loss + self.domain_loss_weight * domain_loss
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(class_logits, dim=1).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            domain_preds.extend(torch.argmax(domain_logits, dim=1).detach().cpu().numpy())
            domain_labels_all.extend(domains.cpu().numpy())

            pbar.set_postfix({'loss': loss.item(), 'grl_lambda': lambda_})

        train_loss /= len(self.train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_domain_acc = accuracy_score(domain_labels_all, domain_preds)
        return train_loss, train_acc, train_domain_acc

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_preds, val_probs, val_labels = [], [], []

        with torch.no_grad():
            for images, labels, _domains in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        class_logits, _ = self.model(images)
                        loss = self.class_criterion(class_logits, labels)
                else:
                    class_logits, _ = self.model(images)
                    loss = self.class_criterion(class_logits, labels)

                val_loss += loss.item()
                probs = torch.softmax(class_logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                val_preds.extend(preds)
                val_probs.extend(probs[:, 1])
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(self.val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except Exception:
            val_auc = 0.0

        return val_loss, val_acc, val_auc, val_preds, val_labels, val_probs

    def train(self):
        print(f"\nDANN training start, {self.num_epochs} epochs (domain_loss_weight={self.domain_loss_weight})")
        print(f"Device: {self.device}, AMP: {self.use_amp}")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            train_loss, train_acc, train_domain_acc = self.train_epoch()
            val_loss, val_acc, val_auc, *_ = self.validate()
            self.scheduler.step(val_auc)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_domain_acc'].append(train_domain_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Domain Acc: {train_domain_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                if self.save_dir:
                    os.makedirs(self.save_dir, exist_ok=True)
                    torch.save(self.best_model_state, os.path.join(self.save_dir, 'best_model.pth'))
                    print(f"saved best model (AUC: {val_auc:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping (patience: {self.early_stopping_patience})")
                break

        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        return self.history

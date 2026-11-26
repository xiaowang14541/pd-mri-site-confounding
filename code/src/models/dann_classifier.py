"""
Domain-adversarial DenseNet121 (Ganin & Lempitsky, 2016 style) for PD/HC classification.

Rationale: baseline_scanner showed dataset identity alone predicts PD/HC label with
AUC ~0.98 (3 of 4 datasets are single-class), and LODO validation showed the plain
DenseNet121 classifier collapses to AUC 0.187 (worse than chance) on unseen sites.
This module adds a domain classifier head fed through a gradient-reversal layer (GRL),
trained adversarially against the shared DenseNet121 feature extractor so that the
extractor is pushed toward site-invariant features, while the class head still learns
PD-vs-HC on top of those (hopefully more generalizable) features.
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DANNDenseNet121(nn.Module):
    def __init__(self, num_classes: int = 2, num_domains: int = 3, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        backbone = models.densenet121(pretrained=pretrained)
        self.features = backbone.features
        num_features = backbone.classifier.in_features  # 1024

        self.class_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )

        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_domains)
        )

    def extract_features(self, x):
        feats = self.features(x)
        feats = nn.functional.relu(feats, inplace=True)
        feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
        feats = torch.flatten(feats, 1)
        return feats

    def forward(self, x):
        feats = self.extract_features(x)
        class_logits = self.class_head(feats)
        domain_logits = self.domain_head(self.grl(feats))
        return class_logits, domain_logits

    def set_grl_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)


def get_dann_model(num_classes: int = 2, num_domains: int = 3, pretrained: bool = True, dropout: float = 0.5):
    return DANNDenseNet121(num_classes=num_classes, num_domains=num_domains, pretrained=pretrained, dropout=dropout)

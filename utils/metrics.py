"""
Generic classification metric helpers that work for both binary and
multiclass tasks. Used by step3 / step6 / step7 / engine evaluation
loops so the same code paths support n_class >= 2 datasets (e.g.
camelyon16 / tcga binary, glioma3 three-class).

`y_pred` is always the post-softmax probability tensor of shape
[N, num_classes]. `y_true` has shape [N] with integer class ids.

For multiclass:
    - AUROC  : macro-OVR AUROC on the full prob matrix.
    - F1     : macro F1 over predicted argmax labels.
    - P/R    : macro Precision / Recall over predicted argmax labels.
    - Acc    : multiclass top-1 accuracy.

For binary (n_class == 2) we fall back to the original single-class
formulation so existing checkpoints and reported numbers stay
backwards compatible.
"""

from typing import Dict

import torch
import torchmetrics
from sklearn.metrics import balanced_accuracy_score


def _is_binary(num_classes: int) -> bool:
    return int(num_classes) == 2


def _common_kwargs(num_classes: int) -> Dict:
    if _is_binary(num_classes):
        return {'task': 'binary'}
    return {'task': 'multiclass', 'num_classes': int(num_classes), 'average': 'macro'}


def compute_auroc(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    """Compute AUROC. Binary uses positive-class probability column,
    multiclass uses the full probability matrix with macro averaging."""
    device = y_pred.device
    if _is_binary(num_classes):
        metric = torchmetrics.AUROC(task='binary').to(device)
        metric(y_pred[:, 1], y_true)
    else:
        metric = torchmetrics.AUROC(
            task='multiclass', num_classes=int(num_classes), average='macro'
        ).to(device)
        metric(y_pred, y_true)
    return metric.compute().item()


def compute_accuracy(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    device = y_pred_labels.device
    metric = torchmetrics.Accuracy(**_common_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().item()


def compute_f1(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    device = y_pred_labels.device
    metric = torchmetrics.F1Score(**_common_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().item()


def compute_precision(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    device = y_pred_labels.device
    metric = torchmetrics.Precision(**_common_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().item()


def compute_recall(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    device = y_pred_labels.device
    metric = torchmetrics.Recall(**_common_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().item()


def compute_balanced_accuracy(y_pred_labels: torch.Tensor, y_true: torch.Tensor) -> float:
    return balanced_accuracy_score(y_true.cpu().numpy(), y_pred_labels.cpu().numpy())


def compute_classification_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """One-shot convenience helper used by every evaluation loop."""
    y_pred_labels = torch.argmax(y_pred, dim=-1)

    return {
        'accuracy': compute_accuracy(y_pred_labels, y_true, num_classes),
        'auroc': compute_auroc(y_pred, y_true, num_classes),
        'f1': compute_f1(y_pred_labels, y_true, num_classes),
        'precision': compute_precision(y_pred_labels, y_true, num_classes),
        'recall': compute_recall(y_pred_labels, y_true, num_classes),
        'balanced_accuracy': compute_balanced_accuracy(y_pred_labels, y_true),
    }

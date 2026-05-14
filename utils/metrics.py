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

Per-class helpers (compute_per_class_*) expose the same numbers but
without averaging, so callers can report a row-per-class table.
"""

from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Per-class helpers (no averaging)
# ---------------------------------------------------------------------------
def _per_class_kwargs(num_classes: int) -> Dict:
    """torchmetrics kwargs that disable averaging so we get one value per
    class for both binary and multiclass tasks."""
    nc = int(num_classes)
    if nc < 2:
        raise ValueError(f"num_classes must be >= 2, got {nc}")
    return {'task': 'multiclass', 'num_classes': nc, 'average': None}


def compute_per_class_precision(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = y_pred_labels.device
    metric = torchmetrics.Precision(**_per_class_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().detach().cpu()


def compute_per_class_recall(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = y_pred_labels.device
    metric = torchmetrics.Recall(**_per_class_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().detach().cpu()


def compute_per_class_f1(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = y_pred_labels.device
    metric = torchmetrics.F1Score(**_per_class_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().detach().cpu()


def compute_per_class_accuracy(y_pred_labels: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-class accuracy. Equivalent to per-class recall in a multiclass
    setting (TP / (TP + FN)), exposed under both names for clarity."""
    device = y_pred_labels.device
    metric = torchmetrics.Accuracy(**_per_class_kwargs(num_classes)).to(device)
    metric(y_pred_labels, y_true)
    return metric.compute().detach().cpu()


def compute_per_class_support(y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Number of ground-truth samples per class."""
    nc = int(num_classes)
    counts = torch.zeros(nc, dtype=torch.long)
    y_true_cpu = y_true.detach().cpu().long().view(-1)
    for c in range(nc):
        counts[c] = int((y_true_cpu == c).sum().item())
    return counts


def compute_per_class_classification_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Return a dict keyed by class name with per-class precision / recall /
    f1 / accuracy / support. Useful for printing a per-class breakdown next
    to the aggregate metrics returned by `compute_classification_metrics`.
    """
    nc = int(num_classes)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(nc)]
    if len(class_names) != nc:
        raise ValueError(
            f"class_names has {len(class_names)} entries but num_classes={nc}"
        )

    y_pred_labels = torch.argmax(y_pred, dim=-1)

    precision = compute_per_class_precision(y_pred_labels, y_true, nc)
    recall = compute_per_class_recall(y_pred_labels, y_true, nc)
    f1 = compute_per_class_f1(y_pred_labels, y_true, nc)
    accuracy = compute_per_class_accuracy(y_pred_labels, y_true, nc)
    support = compute_per_class_support(y_true, nc)

    return {
        class_names[i]: {
            'precision': float(precision[i].item()),
            'recall': float(recall[i].item()),
            'f1': float(f1[i].item()),
            'accuracy': float(accuracy[i].item()),
            'support': int(support[i].item()),
        }
        for i in range(nc)
    }


def format_per_class_metrics_table(
    per_class_metrics: Dict[str, Dict[str, float]],
    title: str = "Per-class metrics",
) -> str:
    """Format the dict returned by `compute_per_class_classification_metrics`
    into a printable table string."""
    header = f"{'Class':<16} | {'Precision':<9} | {'Recall':<6} | {'F1':<6} | {'Accuracy':<8} | {'Support':<7}"
    sep = "-" * len(header)
    lines = [title, header, sep]
    for cls_name, m in per_class_metrics.items():
        lines.append(
            f"{cls_name:<16} | {m['precision']:.4f}    | {m['recall']:.4f} | {m['f1']:.4f} | {m['accuracy']:.4f}   | {m['support']:<7d}"
        )
    return "\n".join(lines)

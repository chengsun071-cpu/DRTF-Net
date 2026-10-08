"""Selected evaluation functions extracted from the original DRTF-Net code.

Operates on pre-computed labels, predictions, and class probabilities.
No raw data, preprocessing, training, or model predictions are supplied.
"""
from __future__ import annotations
import json
from typing import Dict, Optional, Sequence
import numpy as np
import torch
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    average_precision_score = roc_auc_score = None

def make_confusion_matrix(num_classes: int) -> torch.Tensor:
    if int(num_classes) < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    return torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long)


def update_confusion_matrix(
    labels: torch.Tensor,
    preds: torch.Tensor,
    confusion: torch.Tensor,
) -> None:
    """Accumulate a CxC confusion matrix on CPU without storing every prediction."""
    num_classes = int(confusion.shape[0])
    labels_cpu = labels.detach().to("cpu", dtype=torch.long).reshape(-1)
    preds_cpu = preds.detach().to("cpu", dtype=torch.long).reshape(-1)

    valid = (
        (labels_cpu >= 0)
        & (labels_cpu < num_classes)
        & (preds_cpu >= 0)
        & (preds_cpu < num_classes)
    )
    if not bool(valid.all()):
        bad_labels = labels_cpu[~valid][:10].tolist()
        bad_preds = preds_cpu[~valid][:10].tolist()
        raise ValueError(
            f"Found labels/predictions outside [0, {num_classes - 1}]. "
            f"labels={bad_labels}, preds={bad_preds}"
        )

    flat = labels_cpu * num_classes + preds_cpu
    confusion += torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def metrics_from_confusion(
    confusion: torch.Tensor | np.ndarray | Sequence[Sequence[int]],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float | int | str]:
    cm = np.asarray(
        confusion.detach().cpu().numpy() if isinstance(confusion, torch.Tensor) else confusion,
        dtype=np.int64,
    )
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape={cm.shape}")

    num_classes = int(cm.shape[0])
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match num_classes ({num_classes})"
        )

    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    tp_per_class = np.diag(cm).astype(np.float64)
    precision_per_class = _safe_divide(tp_per_class, predicted)
    recall_per_class = _safe_divide(tp_per_class, support)
    f1_per_class = _safe_divide(
        2.0 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
    )

    total = float(cm.sum())
    acc = float(tp_per_class.sum() / total) if total > 0 else 0.0
    macro_precision = float(precision_per_class.mean()) if num_classes else 0.0
    macro_recall = float(recall_per_class.mean()) if num_classes else 0.0
    macro_f1 = float(f1_per_class.mean()) if num_classes else 0.0
    weighted_f1 = float(np.sum(f1_per_class * support) / total) if total > 0 else 0.0

    # Generalized multiclass MCC (Gorodkin/Jurman formulation).
    row_sums = cm.sum(axis=1).astype(np.float64)
    col_sums = cm.sum(axis=0).astype(np.float64)
    trace = float(np.trace(cm))
    s = float(cm.sum())
    numerator = trace * s - float(np.dot(row_sums, col_sums))
    denominator = np.sqrt(
        max(s * s - float(np.dot(col_sums, col_sums)), 0.0)
        * max(s * s - float(np.dot(row_sums, row_sums)), 0.0)
    )
    mcc = float(numerator / denominator) if denominator > 0 else 0.0

    per_class = {
        str(class_names[i]): {
            "class_index": i,
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }
        for i in range(num_classes)
    }

    metrics: Dict[str, float | int | str] = {
        "acc": acc,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_acc": macro_recall,
        "mcc": mcc,
        "num_classes": num_classes,
        "confusion_matrix": json.dumps(cm.tolist(), ensure_ascii=False),
        "per_class_metrics": json.dumps(per_class, ensure_ascii=False),
        "tn": "",
        "fp": "",
        "fn": "",
        "tp": "",
    }

    # Preserve the old binary fields when C=2 for backward compatibility.
    if num_classes == 2:
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})

    return metrics


def compute_probability_metrics(
    y_true: Sequence[int],
    probability_matrix: Sequence[Sequence[float]] | np.ndarray,
    num_classes: int,
) -> Dict[str, float | str]:
    """Compute macro OvR AUROC and macro AUPRC for binary or multiclass outputs."""
    result: Dict[str, float | str] = {"auroc": "", "auprc": ""}
    if len(y_true) == 0:
        return result

    y = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probability_matrix, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != int(num_classes):
        raise ValueError(
            f"probability_matrix must have shape [N, {num_classes}], got {probs.shape}"
        )

    observed = np.unique(y)
    if observed.size < 2:
        return result

    if roc_auc_score is not None:
        try:
            if int(num_classes) == 2:
                result["auroc"] = float(roc_auc_score(y, probs[:, 1]))
            else:
                result["auroc"] = float(
                    roc_auc_score(
                        y,
                        probs,
                        labels=list(range(int(num_classes))),
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except Exception:
            result["auroc"] = ""

    if average_precision_score is not None:
        try:
            if int(num_classes) == 2:
                result["auprc"] = float(average_precision_score(y, probs[:, 1]))
            else:
                y_one_hot = np.eye(int(num_classes), dtype=np.int64)[y]
                result["auprc"] = float(
                    average_precision_score(y_one_hot, probs, average="macro")
                )
        except Exception:
            result["auprc"] = ""

    return result


def format_confusion_matrix(
    confusion: torch.Tensor | np.ndarray | Sequence[Sequence[int]],
    class_names: Sequence[str],
) -> str:
    cm = np.asarray(
        confusion.detach().cpu().numpy() if isinstance(confusion, torch.Tensor) else confusion,
        dtype=np.int64,
    )
    header = "true\\pred\t" + "\t".join(str(x) for x in class_names)
    rows = [header]
    for i, name in enumerate(class_names):
        rows.append(str(name) + "\t" + "\t".join(str(int(x)) for x in cm[i].tolist()))
    return "\n".join(rows)

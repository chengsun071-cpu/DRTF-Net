"""Synthetic-only metric example, not DRTF-Net paper results."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation_metrics import make_confusion_matrix, update_confusion_matrix, metrics_from_confusion

if __name__ == "__main__":
    y_true = torch.tensor([0, 0, 1, 1, 2, 2])
    y_pred = torch.tensor([0, 1, 1, 1, 2, 0])
    cm = make_confusion_matrix(num_classes=3)
    update_confusion_matrix(y_true, y_pred, cm)
    result = metrics_from_confusion(cm)
    print("Synthetic example, NOT a model evaluation:")
    print("Accuracy: {:.4f}".format(result["acc"]))
    print("Macro F1: {:.4f}".format(result["macro_f1"]))

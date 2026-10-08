"""Synthetic-only forward pass of a *partial* fusion component."""
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.fusion_head import FusionClassifierHead

if __name__ == "__main__":
    torch.manual_seed(0)
    model = FusionClassifierHead(graph_out_dim=128, fusion_hidden_dim=256,
                                 clf_hidden_dim=256, num_classes=3, dropout=0.2)
    model.eval()
    with torch.no_grad():
        # Random vectors, not embeddings computed from traffic.
        a = torch.randn(4, 128)
        b = torch.randn(4, 128)
        logits = model(a, b)
    print("Synthetic fusion-only logits shape:", tuple(logits.shape))
    print("This is not the DRTF-Net end-to-end model or an accuracy test.")

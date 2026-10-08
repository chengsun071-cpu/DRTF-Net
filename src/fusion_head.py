"""Selected fusion/classification layers from original model.py.

This *partial* neural network consumes already prepared branch embeddings.
No PCAP processing, graph building, flow-relation learning, or training pipeline
is included. The public class below is not the complete paper model.
"""
from __future__ import annotations
import torch
from torch import nn

class FusionClassifierHead(nn.Module):
    """Fuse two ready-to-use branch embeddings into class logits."""

    def __init__(self, graph_out_dim: int, fusion_hidden_dim: int,
                 clf_hidden_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        if min(graph_out_dim, fusion_hidden_dim, clf_hidden_dim) <= 0 or num_classes < 2:
            raise ValueError("Invalid dimensions or number of classes")
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(graph_out_dim * 2),
            nn.Linear(graph_out_dim * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, graph_out_dim * 2),
            nn.LayerNorm(graph_out_dim * 2),
        )
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(graph_out_dim * 2),
            nn.Linear(graph_out_dim * 2, clf_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden_dim, num_classes),
        )

    def forward(self, packet_embedding: torch.Tensor, flow_embedding: torch.Tensor) -> torch.Tensor:
        if packet_embedding.ndim != 2 or flow_embedding.ndim != 2:
            raise ValueError("Expected [batch_size, embedding_dimension] tensors")
        if packet_embedding.shape != flow_embedding.shape:
            raise ValueError("Both branch embeddings must have identical shapes")
        fused = self.final_fusion(torch.cat([packet_embedding, flow_embedding], dim=1))
        return self.final_classifier(fused)

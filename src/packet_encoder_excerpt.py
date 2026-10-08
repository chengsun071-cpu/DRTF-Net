"""Selected *original* packet-graph encoder components from the DRTF-Net code.

Not a complete DRTF-Net model. The flow graph construction, packet graph
construction, flow relation model, PCAP parsing, features, and training code
are not included. DGL is only required when executing the graph encoder.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

try:
    import dgl
    from dgl.nn import AvgPooling, SAGEConv
    from dgl.nn import EdgeGATConv
except (ImportError, OSError):
    dgl = None
    AvgPooling = SAGEConv = EdgeGATConv = None

def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d = in_dim
    for i in range(num_layers - 1):
        layers += [nn.Linear(d, hidden_dim), nn.SiLU(), nn.Dropout(dropout)]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class EdgeFeatureEncoder(nn.Module):
    """
    将 (edge_type, edge_attr) -> edge_emb （可学习）
    """
    def __init__(
        self,
        num_edge_types: int,
        edge_attr_dim: int,
        edge_type_emb_dim: int,
        edge_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_edge_types = int(num_edge_types)
        self.edge_attr_dim = int(edge_attr_dim)

        self.edge_type_emb = nn.Embedding(self.num_edge_types, edge_type_emb_dim)

        in_dim = edge_type_emb_dim + self.edge_attr_dim
        self.mlp = make_mlp(in_dim, edge_hidden_dim, edge_hidden_dim, dropout=dropout, num_layers=2)
        self.norm = nn.LayerNorm(edge_hidden_dim)

        # 用于把 edge_emb 压缩成标量 edge_weight（给 SAGEConv/GNN 使用）
        self.weight_head = nn.Sequential(
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid(),  # 约束到 (0,1) 便于稳定训练
        )

    def forward(self, g: dgl.DGLGraph, edge_attr_override: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_type = g.edata["edge_type"].long()
        type_emb = self.edge_type_emb(edge_type)

        if self.edge_attr_dim > 0:
            edge_attr = edge_attr_override if edge_attr_override is not None else g.edata["edge_attr"]
            edge_attr = edge_attr.float()
            x = torch.cat([type_emb, edge_attr], dim=1)
        else:
            x = type_emb

        edge_emb = self.mlp(x)
        edge_emb = self.norm(edge_emb)

        edge_weight = self.weight_head(edge_emb).squeeze(-1)
        return edge_emb, edge_weight


class GraphEncoderEdgeGAT(nn.Module):
    """
    Edge-aware Encoder:
    - node_feat + node_type embedding -> node hidden
    - (edge_type + edge_attr) -> edge_emb
    - EdgeGATConv 用 edge_emb 计算注意力与消息传递
    - 额外用 SAGEConv + edge_weight 再走两层（可关）
    """
    def __init__(
        self,
        node_feat_dim: int,
        node_type_emb_dim: int,
        num_edge_types: int,
        edge_attr_dim: int,
        edge_type_emb_dim: int,
        edge_hidden_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        out_dim: int,
        use_post_sage: bool = True,
    ):
        super().__init__()
        if EdgeGATConv is None:
            raise ImportError(
                "EdgeGATConv is not available in your DGL installation. "
                "Please upgrade DGL (>=2.x recommended) or choose model_kind=nnconv/mpnn/baseline_gat."
            )

        self.node_type_emb = nn.Embedding(2, node_type_emb_dim)

        self.in_proj = nn.Linear(node_feat_dim + node_type_emb_dim, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)

        self.edge_encoder = EdgeFeatureEncoder(
            num_edge_types=num_edge_types,
            edge_attr_dim=edge_attr_dim,
            edge_type_emb_dim=edge_type_emb_dim,
            edge_hidden_dim=edge_hidden_dim,
            dropout=dropout,
        )

        # EdgeGATConv: 注意力显式使用 edge_feat
        self.egat1 = EdgeGATConv(
            in_feats=hidden_dim,
            edge_feats=edge_hidden_dim,
            out_feats=hidden_dim,
            num_heads=heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            activation=F.elu,
            allow_zero_in_degree=True,
        )
        self.egat2 = EdgeGATConv(
            in_feats=hidden_dim * heads,
            edge_feats=edge_hidden_dim,
            out_feats=hidden_dim,
            num_heads=1,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            activation=F.elu,
            allow_zero_in_degree=True,
        )

        self.use_post_sage = bool(use_post_sage)
        if self.use_post_sage:
            # 进一步注入 edge_weight（标量），让后续层也受边影响
            self.sage1 = SAGEConv(hidden_dim, hidden_dim, aggregator_type="mean")
            self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type="mean")
        else:
            self.sage1 = None
            self.sage2 = None

        self.dropout = nn.Dropout(dropout)
        self.pool = AvgPooling()
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        g: dgl.DGLGraph,
        *,
        edge_attr_override: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        return_attention=True 时返回 (graph_emb, attention_weights_layer1, attention_weights_layer2)
        """
        x = g.ndata["feat"].float()
        t = g.ndata["node_type"].long()
        t_emb = self.node_type_emb(t)
        h = torch.cat([x, t_emb], dim=1)
        h = self.in_norm(self.in_proj(h))

        edge_emb, edge_weight = self.edge_encoder(g, edge_attr_override=edge_attr_override)

        if return_attention:
            h1, attn1 = self.egat1(g, h, edge_emb, get_attention=True)
        else:
            h1 = self.egat1(g, h, edge_emb, get_attention=False)
            attn1 = None

        h1 = h1.flatten(1)
        h1 = self.dropout(h1)

        if return_attention:
            h2, attn2 = self.egat2(g, h1, edge_emb, get_attention=True)
        else:
            h2 = self.egat2(g, h1, edge_emb, get_attention=False)
            attn2 = None

        h2 = h2.squeeze(1)
        h2 = self.dropout(h2)

        if self.use_post_sage:
            h2 = F.elu(self.sage1(g, h2, edge_weight=edge_weight))
            h2 = self.dropout(h2)
            h2 = F.elu(self.sage2(g, h2, edge_weight=edge_weight))
            h2 = self.dropout(h2)

        hg = self.pool(g, h2)
        hg = self.out_norm(self.out_proj(hg))

        if return_attention:
            return hg, attn1, attn2
        return hg

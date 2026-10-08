# DRTF-Net — Selected Public Code (Partial Release)

**Companion materials; NOT the full experimental source code.**

This GitHub-friendly package contains a **limited subset of actual DRTF-Net model source code** and evaluation utilities, alongside dataset information. It has been prepared as a *partial release* for the manuscript *DRTF-Net: Dual-Graph Fine-Grained Encrypted Traffic Classification with Tri-Relation Inter-Flow Modeling*.

## Included

| Item | Contents |
| --- | --- |
| `src/packet_encoder_excerpt.py` | Extracted packet-side graph encoding components (not the packet graph construction algorithm). Requires PyTorch and DGL when used. |
| `src/fusion_head.py` | Fusion and classifier layers reorganized from the original model into an isolated component. |
| `src/evaluation_metrics.py` | Extracted accuracy, macro F1, precision, recall, MCC and optional AUROC/AUPRC evaluation functions. |
| `examples/metrics_demo.py` | Evaluation with **synthetic labels only**. |
| `examples/fusion_demo.py` | Standalone fusion-head forward pass with **random vectors only**. |
| `docs/DATA_SOURCES.md` | Dataset source descriptions. |
| `docs/METHOD_OVERVIEW.md` | Public high-level overview of the published method. |
| `CODE_AVAILABILITY.md` | Clear disclosure of release limitations. |

## Not included

- **Packet/intra-flow graph construction**: packet and burst-context node/edge generation and actual traffic-derived graph preparation.
- **Inter-flow tri-relation construction or relation-aware flow encoder**: source for computing graph relationships and inter-flow messages is not included.
- **Data processing and training pipeline**: PCAP parsing, feature extraction, labeled dataset construction, graph caching, data splits, experiment configurations and model optimization.
- Original full-model entry point, evaluation prediction files, pretrained checkpoints, processed or raw datasets.

The components in this repository **cannot reproduce** DRTF-Net's experiments, trained model predictions, or reported results. These materials should never be cited as a fully reproducible implementation. See `CODE_AVAILABILITY.md`.

## Requirements and small demonstrations

Python 3.10+ and `torch`, `numpy` are needed for the examples. `scikit-learn` is optional for AUROC/AUPRC. The graph encoder excerpt requires a compatible DGL installation **only if that code is used**; there is no graph-building example here.

From the repository root:

```bash
python examples/metrics_demo.py
python examples/fusion_demo.py
```

These only use synthetic data and **do not** run the encrypted-traffic classification model.

## Before publication

This repository contains a **partial code release by author choice**. PeerJ or another venue may request additional code, data or a justified access restriction for verification. The repository alone does not establish journal policy compliance. Review the contents and any relevant journal requirements before making it public.

## License

No open-source license is provided here. All rights remain with the respective copyright holders. Third-party data are not redistributed.

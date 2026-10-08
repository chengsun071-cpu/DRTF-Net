# DRTF-Net

**DRTF-Net: Fine-grained encrypted traffic classification via intra-flow and inter-flow dual-graph modeling**

DRTF-Net is a graph neural network framework for fine-grained encrypted traffic classification. The framework combines packet-level information within individual traffic flows with relationships among associated flows through a dual-graph architecture.

This repository provides selected model components, evaluation utilities, and supporting research documentation for DRTF-Net.

## Overview

DRTF-Net uses complementary graph representations:

- **Intra-flow modeling:** Captures packet interactions and burst-level context within each flow.
- **Inter-flow modeling:** Describes entity consistency, concurrency, and trigger relationships among associated flows.
- **Feature fusion:** Combines learned representations for fine-grained traffic classification.

## Repository Structure

| Path | Description |
| --- | --- |
| `src/packet_encoder_excerpt.py` | Packet graph encoder components (expects a constructed graph) |
| `src/fusion_head.py` | Feature fusion and classification components |
| `src/evaluation_metrics.py` | Classification evaluation utilities |
| `examples/metrics_demo.py` | Evaluation metrics demonstration |
| `examples/fusion_demo.py` | Feature fusion demonstration |
| `docs/DATA_SOURCES.md` | Dataset background and sources |
| `docs/METHOD_OVERVIEW.md` | Overview of the framework |
| `CODE_AVAILABILITY.md` | Code availability information |

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn (optional, for AUROC/AUPRC)
- DGL (for graph encoder components)

## Usage

From the repository root, run:

```bash
python examples/metrics_demo.py
python examples/fusion_demo.py
```

The examples use synthetic labels or random feature vectors to demonstrate the included components. They are not end-to-end traffic classification experiments.

## Datasets

The associated study uses the following traffic datasets:

- **USTC-TFC2016:** Network traffic classification data covering benign and malicious traffic categories.
- **Malware Capture Facility Project (MCFP):** Traffic captures used to construct a multiclass evaluation task.

Dataset information is provided in `docs/DATA_SOURCES.md`. No third-party traffic data are redistributed in this repository.

## Code Availability

Some core research implementations and experimental processing pipelines are not publicly distributed due to research confidentiality considerations. See [`CODE_AVAILABILITY.md`](CODE_AVAILABILITY.md) for details on the available materials and their scope.

## License

No open-source license is provided. All rights reserved.

# DRTF-Net

**Dual-Graph Fine-Grained Encrypted Traffic Classification with Tri-Relation Inter-Flow Modeling**

DRTF-Net is a graph neural network-based framework for fine-grained encrypted traffic classification. It models intra-flow packet interactions and inter-flow relationships through a dual-graph architecture to learn complementary representations of encrypted network traffic.

This repository provides research code, model components, evaluation utilities, and supporting documentation associated with the DRTF-Net framework.

## Overview

DRTF-Net integrates two complementary graph representations:

- **Intra-flow modeling:** Captures packet-level interactions and burst-level contextual information within individual flows.
- **Inter-flow modeling:** Characterizes relationships among associated flows through entity consistency, concurrency, and trigger relationships.
- **Dual-graph feature fusion:** Combines representations from intra-flow and inter-flow branches for fine-grained traffic classification.

## Repository Structure

| File | Description |
|---|---|
| `src/packet_encoder_excerpt.py` | Packet-level graph encoder components |
| `src/fusion_head.py` | Feature fusion and classification modules |
| `src/evaluation_metrics.py` | Classification evaluation metrics |
| `examples/metrics_demo.py` | Evaluation metrics example |
| `examples/fusion_demo.py` | Feature fusion example |
| `docs/DATA_SOURCES.md` | Dataset information and references |
| `docs/METHOD_OVERVIEW.md` | Overview of the proposed framework |

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn
- DGL (for graph encoder components)

## Usage

Run the evaluation example:

```bash
python examples/metrics_demo.py
```

Run the feature fusion example:

```bash
python examples/fusion_demo.py
```

These examples demonstrate individual components using synthetic inputs.

## Datasets

The research is evaluated on two publicly available traffic datasets:

- **USTC-TFC2016:** Network traffic dataset containing benign and malicious traffic categories.
- **MCFP:** Malware Capture Facility Project traffic captures, used to construct a multiclass traffic classification benchmark.

Dataset descriptions and source information are available in `docs/DATA_SOURCES.md`.

## Code Availability

This repository contains research code and supporting materials for DRTF-Net. Certain core implementation modules and experimental pipelines are not publicly released due to research confidentiality restrictions.

## License

No open-source license is currently provided. All rights reserved.

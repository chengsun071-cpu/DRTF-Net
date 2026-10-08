# Method overview (non-executable)

DRTF-Net uses two complementary graph views for encrypted network traffic classification:

- An intra-flow view encodes packet-level sequence structure and contextual information.
- An inter-flow view represents relations among related flows.
- A fusion component combines the learned views for multiclass classification.

The actual packet graph construction and inter-flow graph/relationship implementations are intentionally not supplied in this partial release. The graph-encoder excerpt in `src/packet_encoder_excerpt.py` processes an **already built** DGL graph; it contains **no** traffic-to-graph conversion. The original end-to-end network and training code are absent.

Please consult the manuscript for the scholarly method description. This companion release is not an executable replication of the full research pipeline.

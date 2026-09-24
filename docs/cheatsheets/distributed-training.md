# Distributed Training

## Primitives

A *rank* identifies a participating process, typically managing one GPU; the *root* is the designated source or destination. All ranks below belong to the participating group.

| Primitive | Communication pattern | Description | Operation type |
| --- | --- | --- | --- |
| **Broadcast** | One-to-many | Copies the **same tensor** from the root to every rank. | `Data Transfer` |
| **Scatter** | One-to-many | Distributes **different chunks** of the root's data to different ranks. | `Data Transfer` |
| **Gather** | Many-to-one | Collects every rank's data at the root, keeping contributions separate. | `Data Transfer` |
| **Reduce** | Many-to-one | Combines tensors element-wise using an operation such as sum or max; only the root receives the result. | `Data Transfer`, `Arithmetic` |
| **All-Reduce** | Many-to-many | Combines tensors element-wise and gives every rank the **same complete result**. | `Data Transfer`, `Arithmetic` |
| **All-Gather** | Many-to-many | Collects every rank's data and gives every rank the **full collection**. | `Data Transfer` |
| **Reduce-Scatter** | Many-to-many | Combines tensors element-wise and gives each rank a **different chunk of the result**. | `Data Transfer`, `Arithmetic` |
| **All-to-All** | Many-to-many | Each rank sends a distinct chunk to every other rank and receives one from each. | `Data Transfer` |
| **Barrier** | All ranks synchronize | Makes each rank wait until every rank in the group reaches the barrier. | — |

*Operation type can have multiple tags. Tags describe the operation on application data; Barrier only synchronizes ranks, so neither applies.*

References: [NCCL collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html), [PyTorch distributed communication](https://docs.pytorch.org/docs/stable/distributed.html).

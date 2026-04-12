# Distributed Computing

Graph algorithms and multi-GPU assembly benchmarks for distributed FEM workflows.

## Scripts

| Script | Description |
|--------|-------------|
| `graph_coloring.py` | Parallel graph coloring on the element adjacency graph |
| `graph_partition.py` | Spectral mesh partitioning into subdomains |
| `benchmark_assembly.py` | Single-GPU vs multi-GPU assembly benchmark using `DistributedMesh` |

## Usage

```bash
python graph_coloring.py       # visualize element graph coloring
python graph_partition.py      # visualize spectral mesh partitioning
python benchmark_assembly.py   # multi-GPU assembly benchmark (auto-detects GPUs)
```

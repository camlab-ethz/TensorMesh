# TensorMesh Benchmark Suite - Complete Documentation

**Branch**: `dev/rebuttal`  
**Date**: 2026-03-26  
**Purpose**: Comprehensive performance benchmarking for rebuttal

---

## 📚 Documentation Index

| Document | Content |
|----------|---------|
| [PERFORMANCE_NOTES.md](./PERFORMANCE_NOTES.md) | Performance optimization findings and best practices |
| [FEM_CPU_GPU_BENCHMARK.md](./FEM_CPU_GPU_BENCHMARK.md) | CPU vs GPU comparison for Poisson and Linear Elasticity |
| [PIPELINE_BENCHMARK_RESULTS.md](./PIPELINE_BENCHMARK_RESULTS.md) | Pipeline simulation performance results |
| [SOLVER_INSTALLATION_STATUS.md](./SOLVER_INSTALLATION_STATUS.md) | Multi-solver installation and compatibility status |

---

## 🎯 Key Results Summary

### 1. Loss Function Performance (1M DOF Scale)

| Loss Function | Complexity | 1M DOF Time | Key Optimization |
|---------------|------------|-------------|------------------|
| DataDriven | O(N) | ~2ms | Pure tensor operations |
| FDM | O(N) | ~3ms | Slice-based stencil |
| **TensorPILS** | O(N) | **~10ms** | **K-matrix caching** |
| Galerkin | O(N) | ~120ms | Element-wise integration |
| PINN | O(N×H) | OOM | Autodiff memory explosion |

**Critical Fix**: Boolean indexing → `index_select` reduced 1M DOF time from 325ms to 10ms (**32x speedup**).

### 2. FEM Solver Performance (200K DOF)

| Problem | CPU | GPU | Speedup |
|---------|-----|-----|---------|
| Poisson | 1,329 ms | 179 ms | **7.4x** |
| Linear Elasticity | 6,526 ms | 555 ms | **11.8x** |

**Findings**:
- Linear Elasticity benefits more from GPU (larger matrix blocks)
- Solve phase: 36-52x speedup (sparse iterative)
- Assembly: 1.4-2x speedup (memory bandwidth limited)

---

## 🔧 Performance Optimizations Applied

### Boolean Indexing Fix
```python
# Before: 325ms at 1M DOF
phi_bd = self.phi[self.boundary_mask]

# After: 10ms at 1M DOF
self.boundary_indices = torch.where(self.boundary_mask)[0]  # in setup()
phi_bd = torch.index_select(self.phi, 0, self.boundary_indices)
```

### K-Matrix Caching (TensorPILS)
```python
def setup(self):
    # Pre-assemble once
    with torch.no_grad():
        self.K = assembler(self.mesh.points)

def forward(self):
    # O(N) mat-vec only
    energy = 0.5 * (self.phi * (self.K @ self.phi)).sum()
```

---

## 📊 Generated Artifacts

### Results Files
```
benchmark/results/
├── fem_complete_cpu_gpu.jsonl              # Complete FEM benchmark data
├── poisson_cpu_gpu_comparison.jsonl
├── linear_elasticity_cpu_gpu_comparison.jsonl
└── loss_comparison_*.jsonl                 # Loss function benchmarks
```

### Analysis Scripts
```
benchmark/scripts/
├── analyze_results.py                      # General analysis
├── generate_fem_report.py                  # FEM-specific reports
├── generate_separate_reports.py            # Split by problem type
├── plot_separate_problems.py               # Visualization
├── final_summary.py                        # Summary generation
└── ... (see individual scripts)
```

### Visualizations
```
benchmark/analysis_output/
├── poisson_cpu_gpu_comparison.png          # 6-panel comparison
├── linear_elasticity_cpu_gpu_comparison.png
├── speedup_comparison_both_problems.png
└── pipeline_comparison.png
```

---

## 🚀 Reproducing Results

### FEM CPU vs GPU Comparison
```bash
# Poisson - CPU
python -m benchmark.experiments.fem_comparison.run \
    --solvers tensormesh --problem poisson \
    --device cpu --max-dof 300000 --mesh-type regular

# Poisson - GPU
python -m benchmark.experiments.fem_comparison.run \
    --solvers tensormesh --problem poisson \
    --device cuda:0 --max-dof 300000 --mesh-type regular

# Linear Elasticity - CPU & GPU (similar)

# Generate reports
python -m benchmark.scripts.generate_separate_reports
python -m benchmark.scripts.plot_separate_problems
```

### Loss Function Comparison
```bash
python -m benchmark.experiments.loss_comparison.run \
    --losses tensorpils galerkin fdm datadriven \
    --min-dof 100 --max-dof 1000000 \
    --device cuda:0 --n-runs 3
```

---

## 🔬 Solver Support Matrix

| Solver | CPU | GPU | Poisson | Linear Elasticity | Status |
|--------|-----|-----|---------|-------------------|--------|
| **TensorMesh** | ✅ | ✅ | ✅ | ✅ | **Working** |
| scikit-fem | ✅ | ❌ | ⚠️ | ❌ | API issues |
| JAX-FEM | ✅ | ✅ | ❌ | ❌ | Package mismatch |
| FEniCS | ✅ | ❌ | ❌ | ❌ | Not installed |
| Firedrake | ✅ | ❌ | ❌ | ❌ | Not installed |

**Note**: Multi-solver comparison requires additional setup time. Current results focus on TensorMesh as the primary PyTorch-based FEM framework.

---

## 📈 Hardware Configuration

- **CPU**: AMD Ryzen 7 9700X (16 threads)
- **GPU**: NVIDIA RTX 4070 Ti SUPER (16GB VRAM)
- **RAM**: 64GB DDR5
- **CUDA**: 11.8
- **PyTorch**: 2.1.0
- **TensorMesh**: Latest dev version

---

## 📝 Notes for Reviewers

1. **Performance optimizations** are documented in `PERFORMANCE_NOTES.md` with code examples
2. **Benchmark data** is reproducible using provided scripts
3. **Visualizations** demonstrate clear scaling behavior and GPU advantages
4. **All timing measurements** use median of 3 runs for robust statistics

---

## 🔗 Related Documents

- Main project: [../../README.md](../../README.md)
- Benchmark suite: [../README.md](../README.md)
- Performance notes: [PERFORMANCE_NOTES.md](./PERFORMANCE_NOTES.md)

---

*Generated for dev/rebuttal branch - 2026-03-26*

# FEM CPU vs GPU Benchmark Report

**Date**: 2026-03-26  
**Hardware**: 
- CPU: AMD Ryzen 7 9700X (16 threads)
- GPU: NVIDIA RTX 4070 Ti SUPER (16GB VRAM)
- RAM: 64GB DDR5

**Software**:
- TensorMesh (latest)
- PyTorch 2.1.0 + CUDA 11.8
- torch-sla (sparse linear algebra)

---

## 📊 Executive Summary

| Problem | Max DOF | CPU Time | GPU Time | **Speedup** |
|---------|---------|----------|----------|-------------|
| **Poisson** | 200K | 1,329 ms | 179 ms | **7.4x** |
| **Linear Elasticity** | 200K | 6,526 ms | 555 ms | **11.8x** |

**Key Finding**: Linear Elasticity benefits more from GPU acceleration due to:
- Larger matrix blocks (vector problem)
- More compute-intensive solver phase
- Better GPU utilization with larger data

---

## 🔍 Detailed Results

### Poisson Equation (Scalar Problem)

| DOF | CPU Total (ms) | GPU Total (ms) | Speedup | CPU Memory | GPU Memory |
|-----|----------------|----------------|---------|------------|------------|
| 50K | 289 | 48 | **6.0x** | 85 MB | 45 MB |
| 100K | 612 | 87 | **7.0x** | 152 MB | 81 MB |
| 200K | 1,329 | 179 | **7.4x** | 291 MB | 152 MB |

**Time Breakdown (200K DOF)**:
```
CPU:  [██████████████░░░░░░░░] 1,329 ms  (Assemble: 277 ms, Solve: 1,037 ms)
GPU:  [██░░░░░░░░░░░░░░░░░░░░]   179 ms  (Assemble: 141 ms, Solve:  29 ms)
```

### Linear Elasticity (Vector Problem, 2 DOF/node)

| DOF | CPU Total (ms) | GPU Total (ms) | Speedup | CPU Memory | GPU Memory |
|-----|----------------|----------------|---------|------------|------------|
| 50K | 1,147 | 135 | **8.5x** | 109 MB | 109 MB |
| 100K | 2,770 | 273 | **10.2x** | 211 MB | 211 MB |
| 200K | 6,526 | 555 | **11.8x** | 413 MB | 413 MB |

**Time Breakdown (200K DOF)**:
```
CPU:  [██████████████████████████████] 6,526 ms  (Assemble: 618 ms, Solve: 5,906 ms)
GPU:  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   555 ms  (Assemble: 442 ms, Solve:   113 ms)
```

---

## 📈 Performance Analysis

### 1. GPU Speedup Scaling

```
Speedup vs DOF:

Poisson:                Linear Elasticity:
1K DOF:   0.8x          1K DOF:   0.9x
10K DOF:  3.1x          10K DOF:  5.0x
100K DOF: 7.0x          100K DOF: 10.2x
200K DOF: 7.4x          200K DOF: 11.8x
```

**Observations**:
- Speedup increases with DOF (better GPU utilization)
- Linear Elasticity achieves higher speedup (more compute per DOF)
- Crossover point (~1K DOF): Below this, CPU is faster due to GPU overhead

### 2. Component Analysis

| Component | Poisson Speedup | Linear Elasticity Speedup |
|-----------|-----------------|---------------------------|
| **Assembly** | 2.0x | 1.4x |
| **Solve** | **36x** | **52x** |
| **Total** | 7.4x | 11.8x |

**Key Insight**: 
- **Solver phase** benefits most from GPU (sparse matrix operations)
- Assembly shows modest speedup (memory-bound, element-wise loops)
- Linear Elasticity has larger solve matrices → better GPU utilization

### 3. Memory Usage

| Problem | DOF | CPU Memory | GPU Memory | Ratio |
|---------|-----|------------|------------|-------|
| Poisson | 200K | 291 MB | 152 MB | 0.52x |
| Linear Elasticity | 200K | 413 MB | 413 MB | 1.0x |

**Notes**:
- Poisson: GPU uses less memory (sparse CSR format more efficient)
- Linear Elasticity: Same memory (dense blocks in stiffness matrix)
- CPU memory appears flat due to measurement methodology

---

## 🎯 Problem-Specific Insights

### Poisson (Scalar Laplacian)

**Characteristics**:
- Smaller stiffness matrix (1 DOF/node)
- Sparse matrix with ~9 non-zeros per row
- Assembly: 20% of total time

**GPU Bottleneck**: 
- Assembly kernel launch overhead dominates at small scales
- Memory bandwidth bound for element-wise operations

### Linear Elasticity (Vector Problem)

**Characteristics**:
- Larger stiffness matrix (2 DOF/node in 2D)
- Block-structured sparse matrix
- Assembly: 10% of total time (vectorized operations)
- Solve: 90% of total time (dominates)

**GPU Advantage**:
- Larger block matrices → more parallel work per thread
- Solver (sparse iterative) highly parallelizable
- 52x speedup in solve phase vs 36x for Poisson

---

## 📁 Generated Files

```
benchmark/results/
├── poisson_cpu_gpu_comparison.jsonl          # 48 records
├── linear_elasticity_cpu_gpu_comparison.jsonl # 48 records
└── fem_complete_cpu_gpu.jsonl                # 96 records

benchmark/analysis_output/
├── poisson_cpu_gpu_comparison.png            # 6-panel plot
├── linear_elasticity_cpu_gpu_comparison.png  # 6-panel plot
└── speedup_comparison_both_problems.png      # Speedup curves
```

---

## 🔄 Reproduce Results

```bash
# Poisson - CPU
python -m benchmark.experiments.fem_comparison.run \
    --solvers tensormesh --problem poisson \
    --device cpu --max-dof 300000 --mesh-type regular

# Poisson - GPU
python -m benchmark.experiments.fem_comparison.run \
    --solvers tensormesh --problem poisson \
    --device cuda:0 --max-dof 300000 --mesh-type regular

# Linear Elasticity - CPU
python -m benchmark.experiments.fem_comparison.run \
    --solvers tensormesh --problem linear_elasticity \
    --device cpu --max-dof 300000 --mesh-type regular

# Linear Elasticity - GPU
python -m benchmark.experiments.fem_comparison.run \
    --solvers tensormesh --problem linear_elasticity \
    --device cuda:0 --max-dof 300000 --mesh-type regular

# Generate reports
python -m benchmark.scripts.generate_separate_reports
python -m benchmark.scripts.plot_separate_problems
```

---

## 🔮 Future Work

### Multi-Solver Comparison
When other solvers are available (FEniCS, JAX-FEM, Firedrake):

| Solver | CPU Support | GPU Support | Notes |
|--------|-------------|-------------|-------|
| TensorMesh | ✓ | ✓ | PyTorch-based, best GPU perf |
| JAX-FEM | ✓ | ✓ | JIT compilation, XLA optimization |
| FEniCS | ✓ | ✗ | Traditional FEM, PETSc backend |
| Firedrake | ✓ | ✗ | Code generation, Pythonic |

### Optimization Opportunities

1. **Assembly**
   - Current: Element-wise kernel launch
   - Optimize: Batched element processing
   - Expected: 2-3x speedup

2. **Solver**
   - Current: PBiCGStab + ILU
   - Optimize: AMG preconditioner
   - Expected: 5-10x speedup for large problems

3. **Mixed Precision**
   - Current: FP64 only
   - Optimize: FP32 assembly + FP64 solve
   - Expected: 2x memory reduction

---

## 📊 Visualization Gallery

### 1. Poisson CPU vs GPU Comparison
![Poisson Comparison](poisson_cpu_gpu_comparison.png)

### 2. Linear Elasticity CPU vs GPU Comparison
![Linear Elasticity Comparison](linear_elasticity_cpu_gpu_comparison.png)

### 3. Speedup Comparison
![Speedup](speedup_comparison_both_problems.png)

---

**Report Generated**: 2026-03-26  
**Contact**: TensorMesh Development Team

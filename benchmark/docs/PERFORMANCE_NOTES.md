# TensorMesh 性能优化笔记

本文档记录 TensorMesh 及损失函数性能优化过程中的关键发现、陷阱和最佳实践。

---

## 📊 性能层级总览

| 损失函数类型 | 时间复杂度 | 1M DOF 典型耗时 | 每 DOF 工作量 | 主要瓶颈 | 适用场景 |
|-------------|-----------|----------------|--------------|---------|---------|
| DataDriven | **O(N)** | ~2ms | 1-2 次运算 | 内存带宽 | 监督学习 |
| FDM | **O(N)** | ~3ms | 9-13 次运算 | 内存带宽 | 有限差分 |
| TensorPILS | **O(N)** | ~10ms | ~9 非零元/行 | 稀疏矩阵乘法 | 物理信息学习 |
| Galerkin | **O(N)** | ~120ms | 单元积分计算 | Python 循环开销 | 传统 FEM 弱形式 |
| PINN | **O(N×H)** | OOM (>16GB) | 网络前向×反向 | 自动微分内存 | 无网格 PINN |

> **N** = 自由度数(DOF), **H** = 神经网络隐藏层维度
> 
> **注意**: 虽然都是 O(N)，但**常数因子**差异巨大 (60x+)。DataDriven/FDM 使用纯张量操作，而 Galerkin 有 Python 循环遍历单元。

---

## ⚡ 关键优化案例

### 1. Boolean Indexing 性能陷阱

#### ❌ 问题代码
```python
# 在 1M DOF 时耗时: ~325ms
boundary_mask = self.boundary_mask  # bool tensor [1000000]
phi_bd = self.phi[boundary_mask]    # boolean indexing - 性能悬崖!
```

#### ✅ 优化后代码
```python
# setup() 中预计算边界索引
self.boundary_indices = torch.where(self.boundary_mask)[0]  # [N_bd]

# forward() 中使用 index_select
phi_bd = torch.index_select(self.phi, 0, self.boundary_indices)  # ~10ms
```

#### 🔍 根因分析
- PyTorch 的 boolean indexing `tensor[mask]` 在大规模张量时会导致 **O(N) 内存分配** 和 **内核启动开销**
- 在 1M DOF 时出现性能悬崖 (325ms → 10ms, **32x 加速**)
- `index_select` 使用预计算的整数索引，避免运行时布尔掩码解析

#### 💡 最佳实践
```python
# 1. 在 setup() 中预计算并缓存索引
class MyLoss(BaseLoss):
    def setup(self):
        # 缓存边界索引 (关键!)
        self.boundary_indices = torch.where(self.boundary_mask)[0]
        self.interior_indices = torch.where(~self.boundary_mask)[0]
        
    def forward(self):
        # 2. 使用 index_select 替代 boolean indexing
        phi_bd = torch.index_select(self.phi, 0, self.boundary_indices)
        # 而不是: phi_bd = self.phi[self.boundary_mask]
```

---

### 2. K-Matrix 缓存优化

#### ❌ 问题代码 (TensorPILS 原始实现)
```python
def forward(self):
    # 每次前向都重新组装刚度矩阵 - O(N) 开销!
    K_assembler = LaplaceElementAssembler.from_mesh(self.mesh)
    K = K_assembler(self.mesh.points)  # 500K DOF: ~1573ms
    
    energy = 0.5 * (self.phi * (K @ self.phi)).sum()
    return energy
```

#### ✅ 优化后代码
```python
def setup(self):
    # 一次性预组装 K 矩阵
    K_assembler = LaplaceElementAssembler.from_mesh(self.mesh)
    with torch.no_grad():
        self.K = K_assembler(self.mesh.points)  # 缓存!
    
def forward(self):
    # 仅执行稀疏 mat-vec - O(N)
    energy = 0.5 * (self.phi * (self.K @ self.phi)).sum()  # 500K DOF: ~5ms
    return energy
```

#### 📈 优化效果
| DOF | 优化前 | 优化后 | 加速比 |
|-----|--------|--------|--------|
| 100K | 89ms | 3ms | **30x** |
| 500K | 1573ms | 5ms | **315x** |
| 1M | OOM | 10ms | **∞** |

#### 💡 最佳实践
```python
# 分离 "一次性设置" 和 "重复计算"
class EfficientLoss(BaseLoss):
    def setup(self):
        """一次性计算，在训练开始前调用"""
        with torch.no_grad():
            # 预组装所有常量矩阵
            self.K = assemble_stiffness_matrix(self.mesh)
            self.M = assemble_mass_matrix(self.mesh)
            # 预计算索引
            self.boundary_indices = torch.where(self.boundary_mask)[0]
    
    def forward(self):
        """每次迭代只执行必要的计算"""
        # 复用预计算的矩阵
        return compute_energy(self.phi, self.K)
```

---

### 3. PINN 内存爆炸问题

#### ❌ 问题代码
```python
def forward(self):
    u = self.network(self.points).squeeze()
    
    # 第一次自动微分
    grad_u = torch.autograd.grad(u.sum(), self.points, 
                                  create_graph=True, retain_graph=True)[0]
    
    # 第二次自动微分 (内存爆炸!)
    laplacian = torch.autograd.grad(grad_u.sum(), self.points,
                                    create_graph=True, retain_graph=True)[0]
    # 500K DOF: 需要 44GB+ 显存, OOM
```

#### 🔍 根因分析
- PINN 需要计算二阶导数 (Laplacian)
- 每次 `torch.autograd.grad` 都会**保留计算图**用于后续反向传播
- 对于 N 个点、L 层网络，中间激活值存储: **O(N × L × H)**
- 500K DOF × 3 层 × 64 隐藏维度 × fp32 ≈ **384MB 仅前向传播**
- 二阶导数需要存储梯度梯度: **>44GB 峰值内存**

#### 💡 缓解策略
```python
# 策略 1: 使用 checkpoint 减少内存 (时间换空间)
from torch.utils.checkpoint import checkpoint

class MemoryEfficientPINN(BaseLoss):
    def forward(self):
        u = checkpoint(self.network, self.points)
        # ... 梯度计算

# 策略 2: 分块计算
batch_size = 10000  # 分块处理
laplacian = []
for i in range(0, n_points, batch_size):
    batch_points = self.points[i:i+batch_size]
    batch_lap = compute_laplacian_batch(batch_points)
    laplacian.append(batch_lap)
laplacian = torch.cat(laplacian)

# 策略 3: 使用有限差分替代自动微分 (精度换效率)
def finite_difference_laplacian(u, points, h=1e-3):
    # O(N) 内存而非 O(N×L)
    ...
```

---

### 4. Galerkin 积分优化

#### ⚠️ 当前瓶颈
Galerkin 损失使用单元级积分:
```python
def forward(self):
    residual = 0
    for element in self.mesh.elements:  # 遍历所有单元
        # 单元积分: ∇u·∇v - fv
        residual += integrate_element(element, self.phi)
    return residual
```

#### 🔍 性能特征
- **1M DOF 耗时**: ~120ms (vs TensorPILS ~10ms)
- **原因**: Python 循环遍历单元 + 单元级矩阵组装
- **复杂度**: O(N) 但常数因子大 (10x+ TensorPILS)

#### 💡 潜在优化方向
```python
# 方向 1: 向量化积分 (类似 TensorPILS)
class VectorizedGalerkin(BaseLoss):
    def setup(self):
        # 预计算所有单元的形函数梯度
        self.element_gradients = precompute_gradients(self.mesh)
        
    def forward(self):
        # 向量化计算而非循环
        # ∇u 在所有单元同时计算
        grad_u = torch.einsum('eqi,iq->eq', self.element_gradients, self.phi)
        # ...

# 方向 2: 稀疏矩阵组装缓存 (类似 K-matrix 优化)
# 将单元积分转换为全局稀疏矩阵乘法
```

---

## 🛠️ 性能调试工具

### PyTorch Profiler 使用
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    for _ in range(10):
        loss = loss_fn()
        loss.backward()

# 打印关键统计
print(prof.key_averages().table(
    sort_by="cuda_time_total", 
    row_limit=10
))
```

### 常用诊断命令
```bash
# 查看 GPU 利用率
nvidia-smi dmon -s u

# PyTorch 内存统计
python -c "import torch; print(torch.cuda.memory_summary())"

# 性能分析 (自动识别瓶颈)
python -m benchmark.experiments.loss_comparison.run \
    --losses tensorpils \
    --max-dof 100000 \
    --profile
```

---

## 📋 性能优化检查清单

### 开发新损失函数时
- [ ] 复杂计算移到 `setup()` 而非 `forward()`
- [ ] 避免在 `forward()` 中使用 boolean indexing
- [ ] 预计算并缓存索引 (`torch.where` 在 setup 中)
- [ ] 使用 `torch.no_grad()` 包装预计算代码
- [ ] 测试 100K/500K/1M DOF 的扩展性
- [ ] 检查内存使用是否线性增长

### 代码审查重点
| 反模式 | 检测方法 | 修复方案 |
|-------|---------|---------|
| `tensor[mask]` in forward | grep "\[.*mask" | 改用 `index_select` |
| 矩阵重组装 | 检查 setup 外是否有 assemble | 缓存到 `self.X` |
| 循环遍历单元 | grep "for.*element" | 考虑向量化 |
| `retain_graph=True` | grep "retain_graph" | 评估是否必需 |

---

## 🎯 硬件特定优化

### CUDA GPU (RTX 4070 Ti SUPER 16GB)
```python
# 1. 使用 TF32 加速 matmul (Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True

# 2. 启用 cudnn benchmark
torch.backends.cudnn.benchmark = True

# 3. 使用 pinned memory 加速 CPU->GPU 传输
points = points.pin_memory()

# 4. 大批量时使用 CUDA graphs (静态图优化)
# 适合损失函数计算图固定的场景
```

### CPU (AMD Ryzen 7 9700X)
```python
# 1. 设置线程数
import os
os.environ["OMP_NUM_THREADS"] = "16"
torch.set_num_threads(16)

# 2. 使用 Intel MKL (如可用)
torch.backends.mkl.verbose = True

# 3. 考虑 JIT 编译
@torch.jit.script
def compute_residual(phi, K, f):
    return 0.5 * (phi * (K @ phi)).sum() - (phi * f).sum()
```

---

## 📚 相关文档

- [TensorMesh API 文档](../../docs/API.md)
- [Benchmark 使用指南](../README.md)
- [Firedrake 安装指南](../environments/INSTALL_FIREDRAKE.md)

---

## 📝 版本历史

| 日期 | 优化内容 | 影响 |
|-----|---------|-----|
| 2026-03-25 | Boolean indexing → index_select | 1M DOF: 325ms → 10ms |
| 2026-03-25 | K-matrix 缓存 (TensorPILS) | 500K DOF: 1573ms → 5ms |
| 2026-03-25 | 边界索引预计算 | 消除 1M DOF 性能悬崖 |

---

**维护者**: TensorMesh Team  
**最后更新**: 2026-03-25

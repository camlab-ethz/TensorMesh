# FEM Solver Installation Status

**Date**: 2026-03-26

---

## ✅ Successfully Installed

### 1. TensorMesh
- **Environment**: `tensormesh-bench`
- **CPU**: ✅ Working
- **GPU**: ✅ Working (CUDA)
- **Status**: Primary benchmark solver

### 2. scikit-fem
- **Environment**: `tensormesh-bench`
- **Version**: 12.0.1
- **CPU**: ⚠️ Partial (API issues)
- **GPU**: ❌ Not supported (CPU only library)
- **Status**: Installed but needs API fixes

### 3. JAX-FEM
- **Environment**: `tensormesh-bench`
- **Version**: 0.0.11
- **Dependencies**: JAX 0.9.2, pyfiglet
- **Status**: ❌ API mismatch (different package than expected)
- **Note**: The installed `jax-fem` has different API than solver implementation expects

---

## ❌ Not Installed (Complex)

### FEniCS
- **Status**: Requires separate conda environment
- **Installation**: `conda create -n fenics -c conda-forge fenics`
- **Complexity**: Medium (conflicts with Firedrake)
- **Platform**: Linux/Mac (Windows via WSL)

### Firedrake
- **Status**: Requires separate conda environment + compilation
- **Installation**: 30-60 minutes, compiles PETSc from source
- **Complexity**: High
- **Platform**: Linux/Mac only (no Windows support)

---

## 🔧 Required Fixes

### scikit-fem Solver
**File**: `benchmark/src/solvers/skfem_solver.py`

Issues:
1. Element type selection (quad vs tri) - ✅ Fixed
2. BilinearForm API mismatch - Needs work

```python
# Current (broken):
@skfem.BilinearForm
def load_form(v, w):
    ...

# Need to check scikit-fem 12.x API
```

### JAX-FEM Solver
**File**: `benchmark/src/solvers/jaxfem_solver.py`

Issue: API mismatch
```python
# Expected API (in solver):
from jax_fem import Mesh, Problem, solve

# Actual API (installed package):
# Different module structure
```

---

## 📊 Current Benchmark Capability

| Solver | Poisson CPU | Poisson GPU | Linear Elasticity CPU | Linear Elasticity GPU |
|--------|-------------|-------------|----------------------|----------------------|
| TensorMesh | ✅ | ✅ | ✅ | ✅ |
| scikit-fem | ⚠️ | N/A | ❌ | N/A |
| JAX-FEM | ❌ | ❌ | ❌ | ❌ |
| FEniCS | ❌ | N/A | ❌ | N/A |
| Firedrake | ❌ | N/A | ❌ | N/A |

**Current Focus**: TensorMesh CPU vs GPU comparison (已完成)

---

## 🚀 Next Steps

### Option 1: Fix scikit-fem (Recommended - Quick)
Update solver to match scikit-fem 12.x API:
```bash
# Test if skfem works
python -c "import skfem; print(skfem.__version__)"

# Check API
python -c "from skfem import BilinearForm; help(BilinearForm)"
```

### Option 2: Install FEniCS (Medium effort)
```bash
conda create -n tensormesh-bench-fenics -c conda-forge fenics python=3.10 -y
conda activate tensormesh-bench-fenics
pip install -e .
```

### Option 3: Install Firedrake (High effort, Linux only)
```bash
# Requires Linux/Mac
conda create -n tensormesh-bench-firedrake python=3.10 -y
# Follow Firedrake install docs...
```

### Option 4: Use Alternative JAX-FEM
Check if there's a different JAX FEM library:
```bash
pip uninstall jax-fem
pip install fenicsx-jax  # or other alternatives
```

---

## 📝 Notes

- **scikit-fem**: Pure Python, CPU only, good for validation
- **JAX-FEM**: May need to implement custom wrapper
- **FEniCS**: Traditional FEM, good reference but complex install
- **Firedrake**: Similar to FEniCS, better for research but no GPU

---

**Recommendation**: For immediate multi-solver comparison, focus on fixing scikit-fem integration. For comprehensive comparison, consider setting up FEniCS in WSL or separate Linux environment.

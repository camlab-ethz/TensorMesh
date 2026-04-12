#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RESULTS_DIR="./benchmark_results"
N_RUNS=5

echo "============================================="
echo " TensorMesh vs FEniCSx Benchmark"
echo "============================================="

# Step 1: TensorMesh
echo ""
echo ">>> Step 1: Running TensorMesh benchmark..."
source ~/venvs/tensorgalerkin/bin/activate
module load stack/.2024-04-silent gcc/8.5.0 mesa-glu/9.0.2 2>/dev/null || true
python benchmark_tensormesh.py --output-dir "$RESULTS_DIR" --n-runs "$N_RUNS"

# Step 2: FEniCSx (run each problem separately to avoid PETSc segfaults)
echo ""
echo ">>> Step 2: Running FEniCSx benchmark..."
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate fenics
python benchmark_fenicsx.py --input-dir "$RESULTS_DIR" --n-runs "$N_RUNS" --problem poisson
python benchmark_fenicsx.py --input-dir "$RESULTS_DIR" --n-runs "$N_RUNS" --problem elasticity

# Step 3: Compare
echo ""
echo ">>> Step 3: Comparing results..."
source ~/venvs/tensorgalerkin/bin/activate
python benchmark_compare.py --results-dir "$RESULTS_DIR"

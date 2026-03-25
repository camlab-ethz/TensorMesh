"""Analyze time complexity scaling of losses."""
import json
import pandas as pd
import numpy as np

# Load data
records = []
with open('results/loss_full.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)

# Filter for CUDA and regular mesh
df_cuda = df[(df['device'] == 'cuda:0') & (df['mesh_type'] == 'regular')]

# Group by loss and dof, get median time
grouped = df_cuda.groupby(['loss_name', 'dof']).agg({
    'time_total': 'median',
    'time_forward': 'median',
    'time_backward': 'median'
}).reset_index()

# Print scaling analysis
print('CUDA Time Scaling Analysis (Regular Mesh):')
print('='*60)
for loss in ['galerkin', 'tensorpils', 'fdm', 'pinn', 'datadriven']:
    loss_data = grouped[grouped['loss_name'] == loss].sort_values('dof')
    if len(loss_data) < 2:
        continue
    print(f'\n{loss.upper()}:')
    for i in range(len(loss_data)):
        row = loss_data.iloc[i]
        print(f"  DOF={int(row['dof']):>8}: {row['time_total']*1000:>8.3f} ms")
    
    # Calculate scaling exponents
    print(f"  Scaling analysis:")
    for i in range(1, len(loss_data)):
        prev = loss_data.iloc[i-1]
        curr = loss_data.iloc[i]
        dof_ratio = curr['dof'] / prev['dof']
        time_ratio = curr['time_total'] / prev['time_total']
        # Estimate complexity: time ~ DOF^n -> n = log(time_ratio) / log(dof_ratio)
        if dof_ratio > 1 and time_ratio > 0:
            n = np.log(time_ratio) / np.log(dof_ratio)
            print(f"    {int(prev['dof'])}->{int(curr['dof'])}: O(N^{n:.2f})")

# Check complexity theoretically
print('\n' + '='*60)
print('THEORETICAL COMPLEXITY:')
print('='*60)
print("""
Galerkin:   O(N) per forward - NodeAssembler loops over all elements
TensorPILS: O(N) per forward - Sparse mat-vec (K is pre-assembled structure)
FDM:        O(N) per forward - Local stencil operation
PINN:       O(N) per forward - MLP forward pass
DataDriven: O(N) per forward - Simple vector ops

ISSUE: TensorPILS line 177 re-assembles K matrix EVERY forward pass!
  K = self.K_assembler(self.tm_mesh.points)  # <-- This should be cached!
  
This causes O(N) assembly + O(N) mat-vec = O(N) total but with large constant.
""")

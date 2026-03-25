import json
import pandas as pd
import numpy as np

# Load fixed data
records = []
with open('results/loss_tensorpils_fixed.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)
grouped = df.groupby('dof').agg({'time_total': 'median'}).reset_index()

print('TensorPILS AFTER FIX:')
print('='*50)
for i in range(len(grouped)):
    row = grouped.iloc[i]
    print(f"DOF={int(row['dof']):>8}: {row['time_total']*1000:>8.3f} ms")

print('\nScaling Analysis:')
for i in range(1, len(grouped)):
    prev = grouped.iloc[i-1]
    curr = grouped.iloc[i]
    dof_ratio = curr['dof'] / prev['dof']
    time_ratio = curr['time_total'] / prev['time_total']
    if dof_ratio > 1 and time_ratio > 0:
        n = np.log(time_ratio) / np.log(dof_ratio)
        complexity = f"O(N^{n:.2f})"
        print(f"  {int(prev['dof'])}->{int(curr['dof'])}: {complexity}")

"""
Generate final benchmark summary report.
"""
import pandas as pd

# Load all available results
df = pd.read_json('benchmark/results/fem_complete_cpu_gpu.jsonl', lines=True)

print('='*80)
print('FINAL BENCHMARK SUMMARY')
print('='*80)
print()
print('Available Solvers: TensorMesh (CPU + GPU)')
print('Other Solvers: scikit-fem (installed, needs API fixes), JAX-FEM (API mismatch)')
print()

for problem in ['poisson', 'linear_elasticity']:
    print(f'\n[{problem.upper()}]')
    print('-'*80)
    
    prob_df = df[df['problem'] == problem]
    summary = prob_df.groupby(['library', 'device', 'dof']).agg({
        'time_total': 'median',
        'time_assemble': 'median',
        'time_solve': 'median',
    }).reset_index()
    
    max_dof = summary['dof'].max()
    
    for device in ['cpu', 'cuda:0']:
        data = summary[(summary['device'] == device) & (summary['dof'] == max_dof)]
        if len(data) > 0:
            row = data.iloc[0]
            dev_name = 'GPU' if 'cuda' in device else 'CPU'
            total_ms = row['time_total'] * 1000
            asm_ms = row['time_assemble'] * 1000
            sol_ms = row['time_solve'] * 1000
            print(f'{dev_name}: {total_ms:>8.2f} ms (Assemble: {asm_ms:>6.2f}, Solve: {sol_ms:>6.2f})')
    
    # Calculate speedup
    cpu_data = summary[(summary['device'] == 'cpu') & (summary['dof'] == max_dof)]
    gpu_data = summary[(summary['device'].str.contains('cuda')) & (summary['dof'] == max_dof)]
    
    if len(cpu_data) > 0 and len(gpu_data) > 0:
        speedup = cpu_data['time_total'].values[0] / gpu_data['time_total'].values[0]
        print(f'GPU Speedup: {speedup:.2f}x')

print()
print('='*80)
print('Solver Installation Status')
print('='*80)
print()
print('Installed & Working:')
print('  - TensorMesh (CPU + GPU)')
print()
print('Installed but Needs Fixes:')
print('  - scikit-fem: API compatibility issues')
print('  - JAX-FEM: Different package API than expected')
print()
print('Not Installed:')
print('  - FEniCS: Requires separate conda environment')
print('  - Firedrake: Requires Linux/Mac, 30-60min compilation')

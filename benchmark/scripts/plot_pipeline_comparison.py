"""
Plot Poisson vs Linear Elasticity pipeline comparison.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def plot_pipeline_comparison(results_file, output_dir='benchmark/analysis_output'):
    df = pd.read_json(results_file, lines=True)
    
    # Aggregate by median
    grouped = df.groupby(['problem', 'mesh_type', 'dof']).agg({
        'time_assemble': 'median',
        'time_solve': 'median',
        'time_total': 'median',
        'memory_peak': 'median'
    }).reset_index()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    problems = ['poisson', 'linear_elasticity']
    mesh_types = ['regular', 'unstructured']
    colors = {'poisson': '#1f77b4', 'linear_elasticity': '#ff7f0e'}
    
    # Plot 1: Total Time
    ax = axes[0, 0]
    for problem in problems:
        for mesh_type in mesh_types:
            data = grouped[(grouped['problem'] == problem) & (grouped['mesh_type'] == mesh_type)]
            label = f"{problem} ({mesh_type})"
            linestyle = '-' if mesh_type == 'regular' else '--'
            ax.loglog(data['dof'], data['time_total'] * 1000, 
                     marker='o', linestyle=linestyle, label=label,
                     color=colors[problem], alpha=0.8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title('Total Time vs DOF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Assemble Time
    ax = axes[0, 1]
    for problem in problems:
        for mesh_type in mesh_types:
            data = grouped[(grouped['problem'] == problem) & (grouped['mesh_type'] == mesh_type)]
            label = f"{problem} ({mesh_type})"
            linestyle = '-' if mesh_type == 'regular' else '--'
            ax.loglog(data['dof'], data['time_assemble'] * 1000, 
                     marker='s', linestyle=linestyle, label=label,
                     color=colors[problem], alpha=0.8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Assemble Time (ms)')
    ax.set_title('Assembly Time vs DOF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Solve Time
    ax = axes[0, 2]
    for problem in problems:
        for mesh_type in mesh_types:
            data = grouped[(grouped['problem'] == problem) & (grouped['mesh_type'] == mesh_type)]
            label = f"{problem} ({mesh_type})"
            linestyle = '-' if mesh_type == 'regular' else '--'
            ax.loglog(data['dof'], data['time_solve'] * 1000, 
                     marker='^', linestyle=linestyle, label=label,
                     color=colors[problem], alpha=0.8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title('Solve Time vs DOF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory Usage
    ax = axes[1, 0]
    for problem in problems:
        for mesh_type in mesh_types:
            data = grouped[(grouped['problem'] == problem) & (grouped['mesh_type'] == mesh_type)]
            label = f"{problem} ({mesh_type})"
            linestyle = '-' if mesh_type == 'regular' else '--'
            ax.loglog(data['dof'], data['memory_peak'], 
                     marker='d', linestyle=linestyle, label=label,
                     color=colors[problem], alpha=0.8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage vs DOF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Time Breakdown (500K DOF regular mesh)
    ax = axes[1, 1]
    dofs_to_plot = [10000, 50000, 100000, 500000]
    x_pos = np.arange(len(dofs_to_plot))
    width = 0.35
    
    for i, problem in enumerate(problems):
        assemble_times = []
        solve_times = []
        for dof in dofs_to_plot:
            data = grouped[(grouped['problem'] == problem) & 
                          (grouped['mesh_type'] == 'regular') & 
                          (grouped['dof'] >= dof * 0.9) & 
                          (grouped['dof'] <= dof * 1.1)]
            if len(data) > 0:
                assemble_times.append(data['time_assemble'].values[0] * 1000)
                solve_times.append(data['time_solve'].values[0] * 1000)
            else:
                assemble_times.append(0)
                solve_times.append(0)
        
        ax.bar(x_pos + i * width, assemble_times, width, 
               label=f'{problem} (assemble)', color=colors[problem], alpha=0.7)
        ax.bar(x_pos + i * width, solve_times, width, 
               bottom=assemble_times, label=f'{problem} (solve)', 
               color=colors[problem], alpha=0.4)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Time Breakdown by Component')
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([f'{d//1000}K' for d in dofs_to_plot])
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Speedup (Poisson vs Linear Elasticity)
    ax = axes[1, 2]
    regular_data = grouped[grouped['mesh_type'] == 'regular']
    
    poisson_times = regular_data[regular_data['problem'] == 'poisson'].sort_values('dof')
    elastic_times = regular_data[regular_data['problem'] == 'linear_elasticity'].sort_values('dof')
    
    # Merge on DOF
    merged = pd.merge(poisson_times, elastic_times, on='dof', suffixes=('_poisson', '_elastic'))
    
    speedup_assemble = merged['time_assemble_elastic'] / merged['time_assemble_poisson']
    speedup_solve = merged['time_solve_elastic'] / merged['time_solve_poisson']
    speedup_total = merged['time_total_elastic'] / merged['time_total_poisson']
    
    ax.semilogx(merged['dof'], speedup_total, 'o-', label='Total', linewidth=2)
    ax.semilogx(merged['dof'], speedup_assemble, 's-', label='Assemble', alpha=0.7)
    ax.semilogx(merged['dof'], speedup_solve, '^-', label='Solve', alpha=0.7)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Slowdown Factor (LE / Poisson)')
    ax.set_title('Linear Elasticity vs Poisson\n(>1 means LE is slower)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/pipeline_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved comparison plot to {output_path}')
    
    # Print summary
    print('\n=== Pipeline Performance Summary ===')
    print('\nRegular Mesh - 500K DOF comparison:')
    for problem in problems:
        data = grouped[(grouped['problem'] == problem) & 
                      (grouped['mesh_type'] == 'regular') & 
                      (grouped['dof'] >= 400000)]
        if len(data) > 0:
            row = data.iloc[0]
            print(f'\n{problem}:')
            print(f'  Total:   {row.time_total*1000:>8.2f} ms')
            print(f'  Assemble:{row.time_assemble*1000:>8.2f} ms ({row.time_assemble/row.time_total*100:.1f}%)')
            print(f'  Solve:   {row.time_solve*1000:>8.2f} ms ({row.time_solve/row.time_total*100:.1f}%)')
            print(f'  Memory:  {row.memory_peak:>8.1f} MB')
    
    if len(merged) > 0:
        print(f'\nLinear Elasticity is ~{speedup_total.mean():.1f}x slower than Poisson (avg)')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        plot_pipeline_comparison(sys.argv[1])
    else:
        plot_pipeline_comparison('benchmark/results/fem_combined_pipeline.jsonl')

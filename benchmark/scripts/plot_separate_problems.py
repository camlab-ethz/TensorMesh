"""
Plot separate comparisons for Poisson and Linear Elasticity.
CPU vs GPU for each problem type.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def plot_problem_comparison(df, problem, output_dir='benchmark/analysis_output'):
    """Generate comparison plots for a single problem."""
    
    # Filter data
    prob_df = df[df['problem'] == problem]
    
    if len(prob_df) == 0:
        print(f"No data for {problem}")
        return
    
    # Aggregate
    summary = prob_df.groupby(['device', 'dof']).agg({
        'time_assemble': 'median',
        'time_solve': 'median',
        'time_total': 'median',
        'memory_peak': 'median',
    }).reset_index()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{problem.replace("_", " ").title()} - CPU vs GPU Comparison', 
                 fontsize=16, fontweight='bold')
    
    colors = {'cpu': '#1f77b4', 'cuda:0': '#ff7f0e'}
    markers = {'cpu': 'o', 'cuda:0': 's'}
    
    # Plot 1: Total Time
    ax = axes[0, 0]
    for device in ['cpu', 'cuda:0']:
        data = summary[summary['device'] == device]
        label = 'GPU' if 'cuda' in device else 'CPU'
        ax.loglog(data['dof'], data['time_total'] * 1000,
                 marker=markers[device], color=colors[device],
                 label=label, linewidth=2, markersize=8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Total Time (ms)')
    ax.set_title('Total Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Assembly Time
    ax = axes[0, 1]
    for device in ['cpu', 'cuda:0']:
        data = summary[summary['device'] == device]
        label = 'GPU' if 'cuda' in device else 'CPU'
        ax.loglog(data['dof'], data['time_assemble'] * 1000,
                 marker=markers[device], color=colors[device],
                 label=label, linewidth=2, markersize=8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Assembly Time (ms)')
    ax.set_title('Assembly Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Solve Time
    ax = axes[0, 2]
    for device in ['cpu', 'cuda:0']:
        data = summary[summary['device'] == device]
        label = 'GPU' if 'cuda' in device else 'CPU'
        ax.loglog(data['dof'], data['time_solve'] * 1000,
                 marker=markers[device], color=colors[device],
                 label=label, linewidth=2, markersize=8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title('Solve Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory Usage
    ax = axes[1, 0]
    for device in ['cpu', 'cuda:0']:
        data = summary[summary['device'] == device]
        label = 'GPU' if 'cuda' in device else 'CPU'
        ax.loglog(data['dof'], data['memory_peak'],
                 marker=markers[device], color=colors[device],
                 label=label, linewidth=2, markersize=8)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Speedup vs DOF
    ax = axes[1, 1]
    cpu_data = summary[summary['device'] == 'cpu'].sort_values('dof')
    gpu_data = summary[summary['device'].str.contains('cuda')].sort_values('dof')
    
    # Merge on DOF
    merged = pd.merge(cpu_data, gpu_data, on='dof', suffixes=('_cpu', '_gpu'))
    
    speedup_total = merged['time_total_cpu'] / merged['time_total_gpu']
    speedup_assemble = merged['time_assemble_cpu'] / merged['time_assemble_gpu']
    speedup_solve = merged['time_solve_cpu'] / merged['time_solve_gpu']
    
    ax.semilogx(merged['dof'], speedup_total, 'o-', label='Total', linewidth=2, markersize=8)
    ax.semilogx(merged['dof'], speedup_assemble, 's-', label='Assembly', alpha=0.7)
    ax.semilogx(merged['dof'], speedup_solve, '^-', label='Solve', alpha=0.7)
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3, label='No speedup')
    ax.set_xlabel('DOF')
    ax.set_ylabel('Speedup (CPU time / GPU time)')
    ax.set_title('GPU Speedup')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Time Breakdown (stacked bar at max DOF)
    ax = axes[1, 2]
    max_dof = summary['dof'].max()
    max_data = summary[summary['dof'] == max_dof]
    
    devices = []
    assemble_times = []
    solve_times = []
    
    for device in ['cpu', 'cuda:0']:
        row = max_data[max_data['device'] == device]
        if len(row) > 0:
            devices.append('GPU' if 'cuda' in device else 'CPU')
            assemble_times.append(row['time_assemble'].values[0] * 1000)
            solve_times.append(row['time_solve'].values[0] * 1000)
    
    x = np.arange(len(devices))
    width = 0.6
    
    ax.bar(x, assemble_times, width, label='Assembly', color='#3498db')
    ax.bar(x, solve_times, width, bottom=assemble_times, label='Solve', color='#e74c3c')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Time Breakdown at {int(max_dof)} DOF')
    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotation
    if len(assemble_times) == 2:
        cpu_total = assemble_times[0] + solve_times[0]
        gpu_total = assemble_times[1] + solve_times[1]
        speedup = cpu_total / gpu_total
        ax.text(0.5, cpu_total * 0.9, 
                f'Speedup: {speedup:.2f}x',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/{problem}_cpu_gpu_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    
    # Print summary
    print(f'\n[{problem.upper()} Summary]')
    print(f'  Max DOF tested: {int(max_dof)}')
    if len(merged) > 0:
        max_speedup = speedup_total.max()
        print(f'  Max GPU speedup: {max_speedup:.2f}x')
    
    return output_path


def plot_combined_speedup(df, output_dir='benchmark/analysis_output'):
    """Plot speedup comparison between problems."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'poisson': '#1f77b4', 'linear_elasticity': '#ff7f0e'}
    
    for problem in ['poisson', 'linear_elasticity']:
        prob_df = df[df['problem'] == problem]
        summary = prob_df.groupby(['device', 'dof']).agg({
            'time_total': 'median',
        }).reset_index()
        
        cpu_data = summary[summary['device'] == 'cpu'].sort_values('dof')
        gpu_data = summary[summary['device'].str.contains('cuda')].sort_values('dof')
        
        merged = pd.merge(cpu_data, gpu_data, on='dof', suffixes=('_cpu', '_gpu'))
        speedup = merged['time_total_cpu'] / merged['time_total_gpu']
        
        label = problem.replace('_', ' ').title()
        ax.semilogx(merged['dof'], speedup, 'o-', label=label, 
                   color=colors[problem], linewidth=2, markersize=8)
    
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('DOF', fontsize=12)
    ax.set_ylabel('GPU Speedup (CPU time / GPU time)', fontsize=12)
    ax.set_title('GPU Speedup: Poisson vs Linear Elasticity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{output_dir}/speedup_comparison_both_problems.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')


def main():
    # Load complete dataset
    df = pd.read_json('benchmark/results/fem_complete_cpu_gpu.jsonl', lines=True)
    
    print(f"Loaded {len(df)} records")
    
    # Generate separate plots for each problem
    for problem in ['poisson', 'linear_elasticity']:
        plot_problem_comparison(df, problem)
    
    # Generate combined speedup plot
    plot_combined_speedup(df)
    
    print("\nAll plots generated!")


if __name__ == '__main__':
    main()

"""
Generate publication-quality plots from benchmark results.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_jsonl(filepath):
    """Load JSONL file into DataFrame."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_fem_comparison_bars(df, output_dir):
    """
    Bar plot for FEM comparison showing assemble/solve/total time (using median and IQR).
    """
    # Aggregate data (using median and IQR)
    grouped = df.groupby(['library', 'dof']).agg({
        'time_assemble': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'time_solve': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'time_total': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'memory_peak': 'median'
    }).reset_index()
    
    grouped.columns = ['library', 'dof', 
                       'assemble_median', 'assemble_iqr',
                       'solve_median', 'solve_iqr',
                       'total_median', 'total_iqr',
                       'memory_median']
    
    libraries = grouped['library'].unique()
    n_libs = len(libraries)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Time breakdown by DOF
    ax = axes[0]
    x = np.arange(len(grouped['dof'].unique()))
    width = 0.25
    
    for i, lib in enumerate(libraries):
        lib_data = grouped[grouped['library'] == lib]
        offset = (i - n_libs/2 + 0.5) * width
        
        ax.bar(x + offset, lib_data['assemble_median'], width, 
               label=f'{lib} (assemble)', alpha=0.8)
        ax.bar(x + offset, lib_data['solve_median'], width,
               bottom=lib_data['assemble_median'],
               label=f'{lib} (solve)', alpha=0.8)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Time (s)')
    ax.set_title('FEM Time Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['dof'].unique())
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Total time comparison
    ax = axes[1]
    for lib in libraries:
        lib_data = grouped[grouped['library'] == lib]
        ax.errorbar(lib_data['dof'], lib_data['total_median'], 
                   yerr=lib_data['total_iqr'],
                   marker='o', label=lib, capsize=3)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Solve Time (median + IQR)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage
    ax = axes[2]
    for lib in libraries:
        lib_data = grouped[grouped['library'] == lib]
        ax.plot(lib_data['dof'], lib_data['memory_median'],
               marker='s', label=lib, linewidth=2)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage (median)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'fem_comparison_bars.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_comparison_stacked(df, output_dir):
    """
    STACKED bar plot for Loss comparison.
    Shows forward + backward time stacked together.
    """
    # Filter to regular meshes only for bar plot (consistent DOF)
    df_regular = df[df['mesh_type'] == 'regular'].copy()
    
    if len(df_regular) == 0:
        print("Warning: No regular mesh data for bar plot")
        return
    
    # Aggregate by loss and dof (using median and IQR for robustness)
    grouped = df_regular.groupby(['loss_name', 'dof']).agg({
        'time_forward': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'time_backward': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'time_total': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'memory_peak': 'median'
    }).reset_index()
    
    grouped.columns = ['loss_name', 'dof',
                       'fwd_median', 'fwd_iqr',
                       'bwd_median', 'bwd_iqr',
                       'total_median', 'total_iqr',
                       'memory_median']
    
    losses = sorted(grouped['loss_name'].unique())
    n_losses = len(losses)
    
    # Find common DOFs across all losses (at least 3 losses must have data)
    dof_counts = grouped.groupby('dof')['loss_name'].count()
    valid_dofs = dof_counts[dof_counts >= 3].index.tolist()
    
    if len(valid_dofs) < 2:
        print("Warning: Not enough common DOF points for stacked bar plot")
        return
    
    dofs = sorted(valid_dofs)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: STACKED bar chart - Forward + Backward
    ax = axes[0]
    x = np.arange(len(dofs))
    width = 0.8 / n_losses
    
    # Use distinct colors for each loss
    colors = sns.color_palette("tab10", n_losses)
    
    for i, loss in enumerate(losses):
        loss_data = grouped[grouped['loss_name'] == loss]
        loss_data = loss_data[loss_data['dof'].isin(dofs)].sort_values('dof')
        
        if len(loss_data) == 0:
            continue
            
        # Align with common DOFs (using median)
        fwd_vals = [loss_data[loss_data['dof'] == d]['fwd_median'].values[0] 
                   if d in loss_data['dof'].values else 0 for d in dofs]
        bwd_vals = [loss_data[loss_data['dof'] == d]['bwd_median'].values[0] 
                   if d in loss_data['dof'].values else 0 for d in dofs]
        
        offset = (i - n_losses/2 + 0.5) * width
        
        # Forward time (bottom)
        ax.bar(x + offset, fwd_vals, width,
               label=f'{loss}', color=colors[i], alpha=0.9)
        
        # Backward time (stacked on top)
        ax.bar(x + offset, bwd_vals, width,
               bottom=fwd_vals,
               color=colors[i], alpha=0.5,
               hatch='///', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('DOF', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.set_title('Forward + Backward Time (Stacked)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(d)}' for d in dofs], rotation=45, ha='right')
    ax.set_yscale('log')
    
    # Custom legend showing forward/backward pattern
    from matplotlib.patches import Patch
    legend_elements = []
    for i, loss in enumerate(losses):
        legend_elements.append(Patch(facecolor=colors[i], alpha=0.9, label=f'{loss}'))
    legend_elements.append(Patch(facecolor='gray', alpha=0.5, hatch='///', 
                                 edgecolor='white', label='backward portion'))
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup comparison with total time
    ax = axes[1]
    
    # Normalize to fastest loss at each DOF
    for i, loss in enumerate(losses):
        loss_data = grouped[grouped['loss_name'] == loss]
        loss_data = loss_data[loss_data['dof'].isin(dofs)].sort_values('dof')
        
        if len(loss_data) == 0:
            continue
        
        speedups = []
        dof_vals = []
        for dof in dofs:
            dof_data = grouped[grouped['dof'] == dof]
            baseline = dof_data['total_median'].min()
            loss_row = loss_data[loss_data['dof'] == dof]
            if len(loss_row) > 0:
                speedup = baseline / loss_row['total_median'].values[0]
                speedups.append(speedup)
                dof_vals.append(dof)
        
        if len(speedups) > 0:
            ax.plot(dof_vals, speedups, 'o-', label=loss, linewidth=2, 
                   color=colors[i], markersize=8)
    
    ax.set_xlabel('DOF', fontsize=11)
    ax.set_ylabel('Speedup (vs Fastest)', fontsize=11)
    ax.set_title('Relative Performance (Regular Mesh)', fontsize=12)
    ax.set_xscale('log')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='baseline')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'loss_comparison_stacked.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_comparison_detailed(df, output_dir):
    """
    Detailed comparison with error bars and statistics (using median and IQR).
    """
    grouped = df.groupby(['loss_name', 'dof']).agg({
        'time_forward': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'time_backward': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'time_total': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)],
        'memory_peak': ['median', lambda x: x.quantile(0.75) - x.quantile(0.25)]
    }).reset_index()
    
    grouped.columns = ['loss_name', 'dof',
                       'fwd_median', 'fwd_iqr',
                       'bwd_median', 'bwd_iqr',
                       'total_median', 'total_iqr',
                       'mem_median', 'mem_iqr']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    losses = grouped['loss_name'].unique()
    colors = sns.color_palette("tab10", len(losses))
    
    # Plot 1: Forward time with IQR error bars
    ax = axes[0, 0]
    for i, loss in enumerate(losses):
        data = grouped[grouped['loss_name'] == loss]
        ax.errorbar(data['dof'], data['fwd_median'], yerr=data['fwd_iqr'],
                   marker='o', label=loss, capsize=3, color=colors[i])
    ax.set_xlabel('DOF')
    ax.set_ylabel('Forward Time (s)')
    ax.set_title('Forward Pass Time (median + IQR)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Backward time with IQR error bars
    ax = axes[0, 1]
    for i, loss in enumerate(losses):
        data = grouped[grouped['loss_name'] == loss]
        ax.errorbar(data['dof'], data['bwd_median'], yerr=data['bwd_iqr'],
                   marker='s', label=loss, capsize=3, color=colors[i])
    ax.set_xlabel('DOF')
    ax.set_ylabel('Backward Time (s)')
    ax.set_title('Backward Pass Time (median + IQR)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Total time comparison
    ax = axes[1, 0]
    for i, loss in enumerate(losses):
        data = grouped[grouped['loss_name'] == loss]
        ax.errorbar(data['dof'], data['total_median'], yerr=data['total_iqr'],
                   marker='^', label=loss, capsize=3, color=colors[i], linewidth=2)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Time (median + IQR)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory usage
    ax = axes[1, 1]
    for i, loss in enumerate(losses):
        data = grouped[grouped['loss_name'] == loss]
        ax.errorbar(data['dof'], data['mem_median'], yerr=data['mem_iqr'],
                   marker='d', label=loss, capsize=3, color=colors[i])
    ax.set_xlabel('DOF')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage (median + IQR)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'loss_comparison_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_loss_comparison_by_device(df, output_dir):
    """
    Separate plots for CPU and GPU (CUDA) performance.
    Generates two separate figures.
    """
    # Check if we have device data
    if 'device' not in df.columns:
        print("No device data available")
        return
    
    # Aggregate by loss, dof, and device (using median)
    grouped = df.groupby(['loss_name', 'dof', 'device']).agg({
        'time_forward': 'median',
        'time_backward': 'median',
        'time_total': 'median',
    }).reset_index()
    
    losses = sorted(grouped['loss_name'].unique())
    devices = sorted(grouped['device'].unique())
    
    if len(devices) < 1:
        print("No device data found")
        return
    
    colors = sns.color_palette("tab10", len(losses))
    loss_colors = {loss: colors[i] for i, loss in enumerate(losses)}
    
    # Separate CPU and CUDA data
    cpu_data = grouped[grouped['device'] == 'cpu']
    cuda_data = grouped[grouped['device'].str.contains('cuda', na=False)]
    
    # Plot 1: CPU Performance
    if len(cpu_data) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, loss in enumerate(losses):
            ax = axes[idx]
            loss_cpu = cpu_data[cpu_data['loss_name'] == loss].sort_values('dof')
            
            if len(loss_cpu) > 0:
                # Stacked area: forward + backward (using median)
                ax.fill_between(loss_cpu['dof'], 0, loss_cpu['time_forward'], 
                               alpha=0.7, color=loss_colors[loss], label='forward')
                ax.fill_between(loss_cpu['dof'], loss_cpu['time_forward'], 
                               loss_cpu['time_total'],
                               alpha=0.4, color=loss_colors[loss], 
                               hatch='///', label='backward')
                ax.plot(loss_cpu['dof'], loss_cpu['time_total'], 
                       'o-', color='black', linewidth=1.5, markersize=5)
            
            ax.set_xlabel('DOF', fontsize=10)
            ax.set_ylabel('Time (s)', fontsize=10)
            ax.set_title(f'{loss.upper()} (CPU)', fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Summary subplot - all losses on CPU
        ax = axes[-1]
        for loss in losses:
            loss_cpu = cpu_data[cpu_data['loss_name'] == loss].sort_values('dof')
            if len(loss_cpu) > 0:
                ax.plot(loss_cpu['dof'], loss_cpu['time_total'], 
                       'o-', label=loss, linewidth=2, markersize=6,
                       color=loss_colors[loss])
        
        ax.set_xlabel('DOF', fontsize=10)
        ax.set_ylabel('Total Time (s)', fontsize=10)
        ax.set_title('CPU: All Losses Comparison', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'loss_comparison_cpu.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    # Plot 2: CUDA/GPU Performance
    if len(cuda_data) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, loss in enumerate(losses):
            ax = axes[idx]
            loss_cuda = cuda_data[cuda_data['loss_name'] == loss].sort_values('dof')
            
            if len(loss_cuda) > 0:
                # Stacked area: forward + backward (using median)
                ax.fill_between(loss_cuda['dof'], 0, loss_cuda['time_forward'], 
                               alpha=0.7, color=loss_colors[loss], label='forward')
                ax.fill_between(loss_cuda['dof'], loss_cuda['time_forward'], 
                               loss_cuda['time_total'],
                               alpha=0.4, color=loss_colors[loss], 
                               hatch='///', label='backward')
                ax.plot(loss_cuda['dof'], loss_cuda['time_total'], 
                       'o-', color='black', linewidth=1.5, markersize=5)
            
            ax.set_xlabel('DOF', fontsize=10)
            ax.set_ylabel('Time (s)', fontsize=10)
            ax.set_title(f'{loss.upper()} (CUDA)', fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Summary subplot - all losses on CUDA
        ax = axes[-1]
        for loss in losses:
            loss_cuda = cuda_data[cuda_data['loss_name'] == loss].sort_values('dof')
            if len(loss_cuda) > 0:
                ax.plot(loss_cuda['dof'], loss_cuda['time_total'], 
                       'o-', label=loss, linewidth=2, markersize=6,
                       color=loss_colors[loss])
        
        ax.set_xlabel('DOF', fontsize=10)
        ax.set_ylabel('Total Time (s)', fontsize=10)
        ax.set_title('CUDA: All Losses Comparison', fontsize=11, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'loss_comparison_cuda.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    # Plot 3: Speedup Comparison (if both devices available)
    if len(cpu_data) > 0 and len(cuda_data) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for loss in losses:
            loss_cpu = cpu_data[cpu_data['loss_name'] == loss]
            loss_cuda = cuda_data[cuda_data['loss_name'] == loss]
            
            if len(loss_cpu) > 0 and len(loss_cuda) > 0:
                # Match by DOF
                merged = loss_cpu.merge(loss_cuda, on='dof', suffixes=('_cpu', '_cuda'))
                if len(merged) > 0:
                    speedup = merged['time_total_cpu'] / merged['time_total_cuda']
                    ax.plot(merged['dof'], speedup, 'o-', label=loss, 
                           linewidth=2, markersize=8, color=loss_colors[loss])
        
        ax.set_xlabel('DOF', fontsize=12)
        ax.set_ylabel('Speedup (CPU time / CUDA time)', fontsize=12)
        ax.set_title('CUDA Speedup vs CPU', fontsize=13, fontweight='bold')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='parity')
        ax.set_xscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'loss_speedup_cuda_vs_cpu.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def print_summary_table(df, output_dir):
    """Generate summary statistics table."""
    
    if 'library' in df.columns:
        # FEM comparison summary
        summary = df.groupby('library').agg({
            'time_assemble': 'mean',
            'time_solve': 'mean',
            'time_total': 'mean',
            'memory_peak': 'mean',
            'l2_error': 'mean'
        }).round(4)
        
        print("\n" + "="*80)
        print("FEM COMPARISON SUMMARY")
        print("="*80)
        print(summary.to_string())
        
        # Save to CSV
        summary.to_csv(Path(output_dir) / 'fem_summary.csv')
        
    elif 'loss_name' in df.columns:
        # Loss comparison summary
        summary = df.groupby('loss_name').agg({
            'time_forward': 'median',
            'time_backward': 'median',
            'time_total': 'median',
            'memory_peak': 'median'
        }).round(4)
        
        print("\n" + "="*80)
        print("LOSS COMPARISON SUMMARY")
        print("="*80)
        print(summary.to_string())
        
        # Add speedup analysis
        print("\n" + "-"*80)
        print("SPEEDUP ANALYSIS (relative to slowest)")
        print("-"*80)
        baseline = summary['time_total'].max()
        speedup = baseline / summary['time_total']
        speedup_df = pd.DataFrame({
            'median_time': summary['time_total'],
            'speedup': speedup
        }).round(2)
        print(speedup_df.to_string())
        
        # Save to CSV
        summary.to_csv(Path(output_dir) / 'loss_summary.csv')
        speedup_df.to_csv(Path(output_dir) / 'loss_speedup.csv')


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output-dir", default="./plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {args.input}...")
    df = load_jsonl(args.input)
    print(f"Loaded {len(df)} records")
    
    # Generate summary
    print_summary_table(df, output_dir)
    
    # Generate plots based on experiment type
    if 'library' in df.columns:
        print("\nGenerating FEM comparison plots...")
        plot_fem_comparison_bars(df, output_dir)
        
    elif 'loss_name' in df.columns:
        print("\nGenerating Loss comparison plots...")
        plot_loss_comparison_stacked(df, output_dir)
        plot_loss_comparison_by_device(df, output_dir)
        plot_loss_comparison_detailed(df, output_dir)
    
    print(f"\n[OK] All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

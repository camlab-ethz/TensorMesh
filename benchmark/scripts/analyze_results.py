"""
Analyze and visualize benchmark results.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(file_path: str) -> pd.DataFrame:
    """Load JSONL results into DataFrame."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_fem_comparison(df: pd.DataFrame, output_dir: str):
    """Generate plots for FEM comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: Time vs DOF for each library
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, time_col in enumerate(['time_assemble', 'time_solve', 'time_total']):
        ax = axes[idx]
        
        for library in df['library'].unique():
            lib_data = df[df['library'] == library]
            grouped = lib_data.groupby('dof')[time_col].agg(['mean', 'std']).reset_index()
            
            ax.loglog(grouped['dof'], grouped['mean'], 'o-', label=library)
            ax.fill_between(grouped['dof'], 
                           grouped['mean'] - grouped['std'],
                           grouped['mean'] + grouped['std'],
                           alpha=0.2)
        
        ax.set_xlabel('DOF')
        ax.set_ylabel(f'{time_col.replace("_", " ").title()} (s)')
        ax.set_title(f'{time_col.replace("_", " ").title()} vs DOF')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fem_time_comparison.png', dpi=300)
    plt.close()
    
    # Plot 2: Memory vs DOF
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for library in df['library'].unique():
        lib_data = df[df['library'] == library]
        grouped = lib_data.groupby('dof')['memory_peak'].agg(['mean', 'std']).reset_index()
        
        ax.loglog(grouped['dof'], grouped['mean'], 'o-', label=library)
        ax.fill_between(grouped['dof'],
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       alpha=0.2)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage vs DOF')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fem_memory_comparison.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def plot_loss_comparison(df: pd.DataFrame, output_dir: str):
    """Generate plots for loss comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # Plot 1: Forward time vs DOF
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for loss_name in df['loss_name'].unique():
        loss_data = df[df['loss_name'] == loss_name]
        grouped = loss_data.groupby('dof')['time_forward'].agg(['mean', 'std']).reset_index()
        
        ax.loglog(grouped['dof'], grouped['mean'], 'o-', label=loss_name)
        ax.fill_between(grouped['dof'],
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       alpha=0.2)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Forward Time (s)')
    ax.set_title('Forward Pass Time vs DOF')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_forward_time.png', dpi=300)
    plt.close()
    
    # Plot 2: Backward time vs DOF
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for loss_name in df['loss_name'].unique():
        loss_data = df[df['loss_name'] == loss_name]
        grouped = loss_data.groupby('dof')['time_backward'].agg(['mean', 'std']).reset_index()
        
        ax.loglog(grouped['dof'], grouped['mean'], 'o-', label=loss_name)
        ax.fill_between(grouped['dof'],
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       alpha=0.2)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Backward Time (s)')
    ax.set_title('Backward Pass Time vs DOF')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_backward_time.png', dpi=300)
    plt.close()
    
    # Plot 3: Memory comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for loss_name in df['loss_name'].unique():
        loss_data = df[df['loss_name'] == loss_name]
        grouped = loss_data.groupby('dof')['memory_peak'].agg(['mean', 'std']).reset_index()
        
        ax.loglog(grouped['dof'], grouped['mean'], 'o-', label=loss_name)
        ax.fill_between(grouped['dof'],
                       grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'],
                       alpha=0.2)
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage vs DOF')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_memory.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    # Determine experiment type
    if 'library' in df.columns:
        # FEM comparison
        print("\nFEM Library Comparison:")
        print("-" * 40)
        
        summary = df.groupby('library').agg({
            'time_total': ['mean', 'std', 'min', 'max'],
            'memory_peak': ['mean', 'std'],
            'dof': ['min', 'max']
        }).round(4)
        
        print(summary)
        
    elif 'loss_name' in df.columns:
        # Loss comparison
        print("\nLoss Function Comparison:")
        print("-" * 40)
        
        summary = df.groupby('loss_name').agg({
            'time_forward': ['mean', 'std'],
            'time_backward': ['mean', 'std'],
            'memory_peak': ['mean', 'std'],
            'dof': ['min', 'max']
        }).round(4)
        
        print(summary)
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output-dir", default="./analysis_output",
                       help="Output directory for plots")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading results from {args.input}...")
    df = load_results(args.input)
    print(f"Loaded {len(df)} records")
    
    # Print summary
    print_summary(df)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        
        # Determine experiment type
        if 'library' in df.columns:
            plot_fem_comparison(df, args.output_dir)
        elif 'loss_name' in df.columns:
            plot_loss_comparison(df, args.output_dir)
        else:
            print("Unknown experiment type, skipping plots")


if __name__ == "__main__":
    main()

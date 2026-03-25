"""
Full Loss Comparison Benchmark - CPU vs CUDA, all losses, up to 1e7 DOF.

This script runs comprehensive benchmarks across:
- All available loss functions (galerkin, tensorpils, fdm, pinn, datadriven)
- Both CPU and CUDA devices
- DOF range from 100 to 10,000,000
"""

import subprocess
import sys
import json
from pathlib import Path
import argparse


def run_experiment(losses, min_dof, max_dof, device, n_runs, output_file):
    """Run a single experiment configuration."""
    cmd = [
        sys.executable, "-m", "experiments.loss_comparison.run",
        "--losses"] + losses + [
        "--min-dof", str(min_dof),
        "--max-dof", str(max_dof),
        "--device", device,
        "--n-runs", str(n_runs),
        "--output", str(output_file),
        "--mesh-type", "both"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {device}, DOF {min_dof}-{max_dof}")
    print(f"Losses: {losses}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout for {device} at DOF {max_dof}")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


def merge_results(result_files, output_file):
    """Merge multiple result files into one."""
    all_records = []
    for f in result_files:
        if Path(f).exists():
            with open(f, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        all_records.append(line)
    
    with open(output_file, 'w') as fp:
        for record in all_records:
            fp.write(record + '\n')
    
    print(f"\nMerged {len(all_records)} records into {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Full Loss Comparison")
    parser.add_argument("--max-dof", type=int, default=10000000,
                       help="Maximum DOF (default: 10M)")
    parser.add_argument("--n-runs", type=int, default=3,
                       help="Number of runs per configuration")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"],
                       help="Devices to test")
    parser.add_argument("--losses", nargs="+", 
                       default=["galerkin", "tensorpils", "fdm", "pinn", "datadriven"],
                       help="Loss functions to test")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # DOF ranges - scale up gradually
    # Note: Galerkin/TensorPILS use sparse matrices and may OOM at high DOF
    dof_ranges = [
        (100, 1000),       # Small
        (1000, 10000),     # Medium
        (10000, 100000),   # Large
        (100000, 1000000), # Very large
    ]
    
    # Only add 10M if requested and not using sparse losses
    if args.max_dof >= 10000000:
        dof_ranges.append((1000000, 10000000))
    
    all_result_files = []
    
    # Test each device separately to avoid GPU/CPU conflicts
    for device in args.devices:
        print(f"\n{'#'*60}")
        print(f"# Testing device: {device}")
        print(f"{'#'*60}")
        
        device_clean = device.replace(":", "_")
        
        for min_dof, max_dof in dof_ranges:
            # Skip impossible combinations
            if device == "cpu" and max_dof > 1000000:
                print(f"Skipping CPU at {max_dof} DOF (too slow)")
                continue
            
            output_file = output_dir / f"loss_{device_clean}_{min_dof}_{max_dof}.jsonl"
            
            success = run_experiment(
                args.losses,
                min_dof,
                max_dof,
                device,
                args.n_runs,
                output_file
            )
            
            if success and Path(output_file).exists():
                all_result_files.append(output_file)
            
            # Check if we should continue (don't waste time if small scales fail)
            if not success and max_dof <= 1000:
                print("Basic scales failing, stopping...")
                break
    
    # Merge all results
    if all_result_files:
        final_output = output_dir / "loss_comparison_full.jsonl"
        merge_results(all_result_files, final_output)
        
        # Generate plots
        print("\n" + "="*60)
        print("Generating plots...")
        print("="*60)
        
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        plot_cmd = [
            sys.executable, "scripts/plot_results.py",
            str(final_output),
            "--output-dir", str(plots_dir)
        ]
        
        try:
            subprocess.run(plot_cmd, check=True)
        except Exception as e:
            print(f"Plot generation failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {final_output}")
        print(f"Plots saved to: {plots_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

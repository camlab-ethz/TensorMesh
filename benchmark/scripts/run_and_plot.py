"""
Run experiments and generate plots in one command.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a shell command with progress reporting."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    print(f"✅ Completed: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments and generate visualizations"
    )
    parser.add_argument("--exp", choices=["fem", "loss", "all"], default="all",
                       help="Which experiment to run")
    parser.add_argument("--solvers", default="tensormesh",
                       help="Comma-separated list of solvers for FEM")
    parser.add_argument("--losses", default="galerkin,tensorpils,fdm",
                       help="Comma-separated list of losses")
    parser.add_argument("--min-dof", type=int, default=100)
    parser.add_argument("--max-dof", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--skip-run", action="store_true",
                       help="Skip running experiments, only plot existing results")
    
    args = parser.parse_args()
    
    # Ensure we're in benchmark directory
    benchmark_dir = Path(__file__).parent.parent
    
    results_dir = benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)
    plots_dir = benchmark_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    success = True
    
    # Run FEM comparison
    if args.exp in ("fem", "all") and not args.skip_run:
        result_file = results_dir / "fem_comparison.jsonl"
        cmd = (
            f"cd {benchmark_dir} && "
            f"python -m experiments.fem_comparison.run "
            f"--solvers {args.solvers} "
            f"--min-dof {args.min_dof} "
            f"--max-dof {args.max_dof} "
            f"--device {args.device} "
            f"--n-runs {args.n_runs} "
            f"--output {result_file}"
        )
        success &= run_command(cmd, "FEM Comparison Experiment")
        
        # Generate plots
        if success:
            cmd = (
                f"cd {benchmark_dir} && "
                f"python scripts/plot_results.py {result_file} "
                f"--output-dir {plots_dir}"
            )
            run_command(cmd, "Generate FEM Plots")
            
            cmd = (
                f"cd {benchmark_dir} && "
                f"python scripts/generate_report.py {result_file} "
                f"--plot-dir {plots_dir} "
                f"--output {plots_dir}/fem_report.html"
            )
            run_command(cmd, "Generate FEM HTML Report")
    
    # Run Loss comparison
    if args.exp in ("loss", "all") and not args.skip_run:
        result_file = results_dir / "loss_comparison.jsonl"
        cmd = (
            f"cd {benchmark_dir} && "
            f"python -m experiments.loss_comparison.run "
            f"--losses {args.losses} "
            f"--min-dof {args.min_dof} "
            f"--max-dof {args.max_dof} "
            f"--device {args.device} "
            f"--n-runs {args.n_runs} "
            f"--output {result_file}"
        )
        success &= run_command(cmd, "Loss Comparison Experiment")
        
        # Generate plots
        if success:
            cmd = (
                f"cd {benchmark_dir} && "
                f"python scripts/plot_results.py {result_file} "
                f"--output-dir {plots_dir}"
            )
            run_command(cmd, "Generate Loss Plots")
            
            cmd = (
                f"cd {benchmark_dir} && "
                f"python scripts/generate_report.py {result_file} "
                f"--plot-dir {plots_dir} "
                f"--output {plots_dir}/loss_report.html"
            )
            run_command(cmd, "Generate Loss HTML Report")
    
    # Just plot existing results
    if args.skip_run:
        print("\n📊 Generating plots from existing results...")
        
        if args.exp in ("fem", "all"):
            result_file = results_dir / "fem_comparison.jsonl"
            if result_file.exists():
                cmd = (
                    f"cd {benchmark_dir} && "
                    f"python scripts/plot_results.py {result_file} "
                    f"--output-dir {plots_dir}"
                )
                run_command(cmd, "Generate FEM Plots")
                
                cmd = (
                    f"cd {benchmark_dir} && "
                    f"python scripts/generate_report.py {result_file} "
                    f"--plot-dir {plots_dir} "
                    f"--output {plots_dir}/fem_report.html"
                )
                run_command(cmd, "Generate FEM HTML Report")
        
        if args.exp in ("loss", "all"):
            result_file = results_dir / "loss_comparison.jsonl"
            if result_file.exists():
                cmd = (
                    f"cd {benchmark_dir} && "
                    f"python scripts/plot_results.py {result_file} "
                    f"--output-dir {plots_dir}"
                )
                run_command(cmd, "Generate Loss Plots")
                
                cmd = (
                    f"cd {benchmark_dir} && "
                    f"python scripts/generate_report.py {result_file} "
                    f"--plot-dir {plots_dir} "
                    f"--output {plots_dir}/loss_report.html"
                )
                run_command(cmd, "Generate Loss HTML Report")
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tasks completed successfully!")
        print(f"📁 Plots saved to: {plots_dir}")
        print(f"   - View bar plots: {plots_dir}/*.png")
        print(f"   - View reports: {plots_dir}/*.html")
    else:
        print("⚠️ Some tasks failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

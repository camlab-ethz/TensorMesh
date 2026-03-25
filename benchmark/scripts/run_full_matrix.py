"""
Run complete benchmark matrix: (Poisson, Linear Elasticity) × (CPU, CUDA)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import subprocess
import json
from datetime import datetime

def run_benchmark(problem: str, device: str, max_dof: int = 300000):
    """Run single benchmark."""
    
    print(f"\n{'='*60}")
    print(f"Running: {problem.upper()} on {device.upper()}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, '-m', 'benchmark.experiments.fem_comparison.run',
        '--solvers', 'tensormesh',
        '--problem', problem,
        '--min-dof', '100',
        '--max-dof', str(max_dof),
        '--mesh-type', 'regular',
        '--dimension', '2',
        '--device', device,
        '--n-runs', '3',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"[ERROR] {result.stderr}")
            return None
            
        # Parse output file
        for line in result.stdout.split('\n'):
            if 'Results saved to:' in line:
                return line.split('Results saved to:')[-1].strip()
                
    except Exception as e:
        print(f"[ERROR] {e}")
        return None
    
    return None


def combine_by_problem(files_dict: dict):
    """Combine results by problem."""
    
    for problem, files in files_dict.items():
        print(f"\nCombining {problem} results...")
        
        all_records = []
        for f in files:
            if f and os.path.exists(f):
                with open(f, 'r') as fp:
                    for line in fp:
                        if line.strip():
                            all_records.append(json.loads(line))
        
        if all_records:
            output = f'benchmark/results/{problem}_cpu_gpu_comparison.jsonl'
            with open(output, 'w') as fp:
                for record in all_records:
                    fp.write(json.dumps(record) + '\n')
            print(f"  Saved: {output} ({len(all_records)} records)")


def main():
    problems = ['poisson', 'linear_elasticity']
    devices = ['cpu', 'cuda:0']
    max_dof = 300000  # Reduced for faster testing
    
    results = {p: [] for p in problems}
    
    for problem in problems:
        for device in devices:
            result_file = run_benchmark(problem, device, max_dof)
            if result_file:
                results[problem].append(result_file)
    
    # Combine results
    combine_by_problem(results)
    
    print("\n" + "="*60)
    print("BENCHMARK MATRIX COMPLETE")
    print("="*60)
    for problem, files in results.items():
        print(f"{problem}: {len(files)} files")


if __name__ == '__main__':
    main()

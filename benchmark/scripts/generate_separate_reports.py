"""
Generate separate CPU/GPU comparison reports for Poisson and Linear Elasticity.
"""
import pandas as pd
import json
import sys

def main():
    # Load all 4 result files
    files = [
        'benchmark/results/fem_comparison_20260326_024059.jsonl',  # Poisson CPU
        'benchmark/results/fem_comparison_20260326_024127.jsonl',  # Poisson CUDA
        'benchmark/results/fem_comparison_20260326_024146.jsonl',  # LE CPU
        'benchmark/results/fem_comparison_20260326_024247.jsonl',  # LE CUDA
    ]

    # Combine all
    all_records = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                for line in fp:
                    if line.strip():
                        all_records.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: {f} not found")
            continue

    df = pd.DataFrame(all_records)
    
    print(f"Loaded {len(df)} total records")
    print(f"Problems: {df['problem'].unique().tolist()}")
    print(f"Devices: {df['device'].unique().tolist()}")

    # Separate by problem and save
    for problem in ['poisson', 'linear_elasticity']:
        prob_df = df[df['problem'] == problem]
        if len(prob_df) > 0:
            output = f'benchmark/results/{problem}_cpu_gpu_comparison.jsonl'
            prob_df.to_json(output, orient='records', lines=True)
            print(f'{problem}: {len(prob_df)} records -> {output}')

    # Summary table
    print()
    print('='*80)
    print('CPU vs GPU COMPARISON - TENSORMESH')
    print('='*80)

    summary = df.groupby(['problem', 'device', 'dof']).agg({
        'time_assemble': 'median',
        'time_solve': 'median', 
        'time_total': 'median',
        'memory_peak': 'median'
    }).reset_index()

    for problem in ['poisson', 'linear_elasticity']:
        print(f'\n[{problem.upper()}]')
        print('-'*80)
        
        prob_data = summary[summary['problem'] == problem]
        
        # Header
        header = f"{'DOF':>10} {'Device':>10} {'Assemble':>12} {'Solve':>12} {'Total':>12} {'Speedup':>10}"
        print(header)
        print(f"{'':>10} {'':>10} {'(ms)':>12} {'(ms)':>12} {'(ms)':>12} {'vs CPU':>10}")
        print('-'*80)
        
        # Get max DOF for this problem
        max_dof = prob_data['dof'].max()
        
        for dof in sorted(prob_data['dof'].unique()):
            if dof < 50000:  # Skip small DOF for brevity
                continue
                
            dof_data = prob_data[prob_data['dof'] == dof]
            
            cpu_time = None
            
            for _, row in dof_data.iterrows():
                device = row['device']
                total_ms = row['time_total'] * 1000
                
                if device == 'cpu':
                    cpu_time = row['time_total']
                    speedup = '-'
                else:
                    if cpu_time:
                        speedup = f'{cpu_time/row["time_total"]:.2f}x'
                    else:
                        speedup = '-'
                
                print(f"{int(dof):>10} {device:>10} {row['time_assemble']*1000:>11.2f} {row['time_solve']*1000:>11.2f} {total_ms:>11.2f} {speedup:>10}")
        
        # Calculate speedup at max DOF
        max_data = prob_data[prob_data['dof'] == max_dof]
        cpu_row = max_data[max_data['device'] == 'cpu']
        gpu_row = max_data[max_data['device'].str.contains('cuda')]
        
        if len(cpu_row) > 0 and len(gpu_row) > 0:
            cpu_time = cpu_row['time_total'].values[0]
            gpu_time = gpu_row['time_total'].values[0]
            speedup = cpu_time / gpu_time
            print('-'*80)
            print(f'GPU Speedup at {int(max_dof)} DOF: {speedup:.2f}x')
    
    # Save complete summary
    print()
    print('='*80)
    print('SAVING COMPLETE DATASET')
    print('='*80)
    df.to_json('benchmark/results/fem_complete_cpu_gpu.jsonl', orient='records', lines=True)
    print('Saved: benchmark/results/fem_complete_cpu_gpu.jsonl')


if __name__ == '__main__':
    main()

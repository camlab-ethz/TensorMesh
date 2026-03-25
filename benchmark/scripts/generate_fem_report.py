"""
Generate FEM pipeline performance report.
"""
import pandas as pd
import numpy as np
import sys

def generate_report(results_file):
    # Load combined results
    df = pd.read_json(results_file, lines=True)

    # Create summary table
    print('=' * 100)
    print('TENSORMESH PIPELINE PERFORMANCE SUMMARY (CUDA)')
    print('=' * 100)
    print()
    print(f"Device: {df['device'].iloc[0]}")
    print(f"Total runs: {len(df)}")
    print()

    # Group by problem, mesh_type, dof
    summary = df.groupby(['problem', 'mesh_type', 'dof']).agg({
        'time_assemble': ['median', 'std'],
        'time_solve': ['median', 'std'],
        'time_total': ['median', 'std'],
        'memory_peak': ['median', 'max'],
    }).reset_index()

    # Flatten column names
    summary.columns = ['problem', 'mesh_type', 'dof', 
                       'assemble_median', 'assemble_std',
                       'solve_median', 'solve_std',
                       'total_median', 'total_std',
                       'memory_median', 'memory_max']

    # Print formatted table
    for problem in ['poisson', 'linear_elasticity']:
        print(f'\n[{problem.upper()}]')
        print('-' * 100)
        print(f'{"Mesh":<12} {"DOF":>10} {"Assemble":>12} {"Solve":>12} {"Total":>12} {"Memory":>12}')
        print(f'{"":<12} {"":>10} {"(ms)":>12} {"(ms)":>12} {"(ms)":>12} {"(MB)":>12}')
        print('-' * 100)
        
        prob_data = summary[summary['problem'] == problem]
        for _, row in prob_data.iterrows():
            print(f'{row.mesh_type:<12} {int(row.dof):>10} '
                  f'{row.assemble_median*1000:>11.2f} '
                  f'{row.solve_median*1000:>11.2f} '
                  f'{row.total_median*1000:>11.2f} '
                  f'{row.memory_median:>11.1f}')
        print('-' * 100)

    # Scaling analysis
    print()
    print('=' * 100)
    print('SCALING ANALYSIS')
    print('=' * 100)

    for problem in ['poisson', 'linear_elasticity']:
        print(f'\n[{problem.upper()}]')
        
        # Regular mesh only for scaling analysis
        reg_data = summary[(summary['problem'] == problem) & (summary['mesh_type'] == 'regular')]
        reg_data = reg_data.sort_values('dof')
        
        if len(reg_data) >= 2:
            # Calculate scaling exponent
            dofs = reg_data['dof'].values.astype(float)
            times = reg_data['total_median'].values.astype(float)
            
            # Skip first few small sizes (warmup/noise)
            if len(dofs) >= 4:
                dofs = dofs[2:]  # Skip smallest
                times = times[2:]
            
            log_dofs = np.log(dofs)
            log_times = np.log(times)
            
            # Linear regression
            coeffs = np.polyfit(log_dofs, log_times, 1)
            scaling_exp = coeffs[0]
            
            print(f'  Scaling exponent (Regular mesh): {scaling_exp:.2f}')
            print(f'  Expected: ~1.0 (optimal O(N) scaling)')
            status = "Good" if 0.8 <= scaling_exp <= 1.3 else "Check"
            print(f'  Status: [{status}]')
            
            # Show specific DOF points
            print(f'\n  Detailed timing (Regular mesh):')
            for _, row in reg_data.iterrows():
                solve_ms = row.solve_median * 1000
                print(f'    DOF {int(row.dof):>9}: {row.total_median*1000:>8.2f} ms '
                      f'(assemble: {row.assemble_median*1000:>6.2f}, solve: {solve_ms:>6.2f})')

    # Compare problems
    print()
    print('=' * 100)
    print('PROBLEM COMPARISON (Regular Mesh, 500K DOF)')
    print('=' * 100)
    
    for problem in ['poisson', 'linear_elasticity']:
        data = summary[(summary['problem'] == problem) & 
                       (summary['mesh_type'] == 'regular') &
                       (summary['dof'] >= 400000)]
        if len(data) > 0:
            row = data.iloc[0]
            print(f'\n{problem}:')
            print(f'  Total time: {row.total_median*1000:.2f} ms')
            print(f'  Assemble:   {row.assemble_median*1000:.2f} ms ({row.assemble_median/row.total_median*100:.1f}%)')
            print(f'  Solve:      {row.solve_median*1000:.2f} ms ({row.solve_median/row.total_median*100:.1f}%)')
            print(f'  Memory:     {row.memory_median:.1f} MB')

    print()
    print('=' * 100)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        generate_report(sys.argv[1])
    else:
        generate_report('benchmark/results/fem_combined_pipeline.jsonl')

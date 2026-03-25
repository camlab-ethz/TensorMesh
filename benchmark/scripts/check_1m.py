import json
import pandas as pd

records = []
with open('results/loss_final_cuda.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)
df_cuda = df[(df['device'] == 'cuda:0') & (df['mesh_type'] == 'regular')]

# Check 1M DOF data
print('Data at 1M DOF:')
print('='*60)
df_1m = df_cuda[df_cuda['dof'] == 1000000]
for loss in df_1m['loss_name'].unique():
    loss_data = df_1m[df_1m['loss_name'] == loss]
    print(f"\n{loss.upper()}:")
    print(f"  forward:  {loss_data['time_forward'].median()*1000:>8.3f} ms")
    print(f"  backward: {loss_data['time_backward'].median()*1000:>8.3f} ms")
    print(f"  total:    {loss_data['time_total'].median()*1000:>8.3f} ms")
    print(f"  memory:   {loss_data['memory_peak'].median():>8.1f} MB")
    
# Check scaling from 500K to 1M
print('\n' + '='*60)
print('Scaling 500K -> 1M:')
print('='*60)
for loss in ['tensorpils', 'galerkin', 'fdm', 'datadriven']:
    loss_data = df_cuda[df_cuda['loss_name'] == loss].sort_values('dof')
    data_500k = loss_data[loss_data['dof'] == 501264]
    data_1m = loss_data[loss_data['dof'] == 1000000]
    if len(data_500k) > 0 and len(data_1m) > 0:
        t_500k = data_500k['time_total'].median()
        t_1m = data_1m['time_total'].median()
        ratio = t_1m / t_500k
        print(f"{loss:>12}: {t_500k*1000:>6.2f}ms -> {t_1m*1000:>6.2f}ms (x{ratio:.2f})")

# Check individual runs for TensorPILS at 1M
print('\n' + '='*60)
print('TensorPILS individual runs at 1M DOF:')
print('='*60)
tensorpils_1m = df_cuda[(df_cuda['loss_name'] == 'tensorpils') & (df_cuda['dof'] == 1000000)]
for idx, row in tensorpils_1m.iterrows():
    print(f"  Run {row['run_id']}: fwd={row['time_forward']*1000:.3f}ms, bwd={row['time_backward']*1000:.3f}ms, total={row['time_total']*1000:.3f}ms")

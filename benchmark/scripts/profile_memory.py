"""Profile GPU memory usage during loss computation."""
import torch
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../../..')

from losses import create_loss
from mesh.generators import MeshConfig, generate_mesh
import numpy as np

def profile_loss(loss_name, dof, device='cuda:0'):
    """Profile memory usage of a loss function."""
    config = MeshConfig(dof=dof, mesh_type='regular', element_type='quad', dimension=2)
    mesh = generate_mesh(config)
    
    def source_fn(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    
    # Reset GPU
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure baseline
    baseline = torch.cuda.memory_allocated() / 1024**2
    
    # Create loss
    loss_obj = create_loss(loss_name, mesh, source_fn, device=device)
    loss_obj.setup()
    
    after_setup = torch.cuda.max_memory_allocated() / 1024**2
    
    # Forward pass
    loss_obj.zero_grad()
    _ = loss_obj.forward()
    
    after_forward = torch.cuda.max_memory_allocated() / 1024**2
    
    # Backward pass
    loss_obj.backward()
    
    after_backward = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"\n{loss_name.upper()} @ {dof} DOF:")
    print(f"  Baseline:     {baseline:.1f} MB")
    print(f"  After setup:  {after_setup:.1f} MB (+{after_setup-baseline:.1f})")
    print(f"  After forward:{after_forward:.1f} MB (+{after_forward-after_setup:.1f})")
    print(f"  After backward:{after_backward:.1f} MB (+{after_backward-after_forward:.1f})")
    
    return {
        'loss': loss_name,
        'dof': dof,
        'setup_mb': after_setup - baseline,
        'forward_mb': after_forward - after_setup,
        'backward_mb': after_backward - after_forward,
        'total_mb': after_backward - baseline
    }

if __name__ == "__main__":
    results = []
    for dof in [100000, 500000, 1000000]:
        for loss in ['tensorpils', 'galerkin']:
            try:
                r = profile_loss(loss, dof)
                results.append(r)
            except Exception as e:
                print(f"Failed {loss} at {dof}: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['loss']:>12} @ {r['dof']:>7}: {r['total_mb']:>6.1f} MB total")

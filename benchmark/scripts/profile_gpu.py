"""
Detailed GPU profiling for loss functions at different scales.
Uses PyTorch Profiler to identify bottlenecks.
"""
import torch
import torch.profiler
import sys
import json
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from losses import create_loss
from mesh.generators import MeshConfig, generate_mesh
import numpy as np


def profile_loss_detailed(loss_name, dof, device='cuda:0', warmup=3, iterations=5):
    """
    Profile a loss function with PyTorch profiler.
    """
    print(f"\n{'='*70}")
    print(f"Profiling {loss_name.upper()} at {dof} DOF")
    print(f"{'='*70}")
    
    # Setup
    config = MeshConfig(dof=dof, mesh_type='regular', element_type='quad', dimension=2)
    mesh = generate_mesh(config)
    
    def source_fn(x):
        return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    
    # Create loss
    loss_obj = create_loss(loss_name, mesh, source_fn, device=device)
    loss_obj.setup()
    
    # Warmup
    torch.cuda.synchronize()
    for _ in range(warmup):
        loss_obj.zero_grad()
        _ = loss_obj.forward()
        loss_obj.backward()
    torch.cuda.synchronize()
    
    # Profile
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(iterations):
            loss_obj.zero_grad()
            with torch.profiler.record_function("forward"):
                loss = loss_obj.forward()
            with torch.profiler.record_function("backward"):
                loss_obj.backward()
            torch.cuda.synchronize()
    
    # Print summary
    print("\nTop 10 CUDA Operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\nTop 10 CPU Operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # Memory stats
    print("\nMemory Summary:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**2:.1f} MB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    
    # Export trace
    trace_file = f"profile_{loss_name}_{dof}.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to: {trace_file}")
    print("Open with chrome://tracing or edge://tracing")
    
    return prof


def compare_scales(loss_name, dofs=[100000, 500000, 1000000]):
    """Compare profiling across different scales."""
    results = []
    
    for dof in dofs:
        try:
            prof = profile_loss_detailed(loss_name, dof)
            
            # Extract key metrics
            cuda_time = sum([e.cuda_time_total for e in prof.key_averages()])
            cpu_time = sum([e.cpu_time_total for e in prof.key_averages()])
            
            results.append({
                'loss': loss_name,
                'dof': dof,
                'cuda_time_ms': cuda_time / 1000,
                'cpu_time_ms': cpu_time / 1000,
                'memory_mb': torch.cuda.max_memory_allocated() / 1024**2
            })
            
            torch.cuda.reset_peak_memory_stats()
            
        except Exception as e:
            print(f"Failed at {dof}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    print("\n" + "="*70)
    print("SCALING ANALYSIS")
    print("="*70)
    for i, r in enumerate(results):
        print(f"\n{r['loss']} @ {r['dof']}:")
        print(f"  CUDA time: {r['cuda_time_ms']:.2f} ms")
        print(f"  CPU time:  {r['cpu_time_ms']:.2f} ms")
        print(f"  Memory:    {r['memory_mb']:.1f} MB")
        
        if i > 0:
            prev = results[i-1]
            time_ratio = r['cuda_time_ms'] / prev['cuda_time_ms']
            dof_ratio = r['dof'] / prev['dof']
            print(f"  Scaling:   O(N^{np.log(time_ratio)/np.log(dof_ratio):.2f})")
    
    return results


if __name__ == "__main__":
    import numpy as np
    
    # Profile TensorPILS at different scales
    print("\n" + "#"*70)
    print("# TENSORPILS PROFILING")
    print("#"*70)
    compare_scales('tensorpils', [100000, 500000, 1000000])
    
    # Profile Galerkin at different scales
    print("\n" + "#"*70)
    print("# GALERKIN PROFILING")
    print("#"*70)
    compare_scales('galerkin', [100000, 500000])

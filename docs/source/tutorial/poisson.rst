Poisson Equation
================

Adaptive Mesh Refinement
------------------------

.. code-block:: python 

    import torch 
    import numpy as np
    from tqdm import tqdm
    from torch_fem import LaplaceElementAssembler, Mesh,  Condenser
    from torch_fem.dataset import PoissonMultiFrequency
    from torch_fem.visualization import StreamPlotter
    import matplotlib.pyplot as plt

    if __name__ == "__main__":
        torch.random.manual_seed(123456)
        mesh      = Mesh.gen_rectangle(chara_length=0.1)
        assembler = LaplaceElementAssembler.from_mesh(mesh)
        equation  = PoissonMultiFrequency(K=8)
        condenser = Condenser(mesh.boundary_mask)

        optimizer = torch.optim.Adam(mesh.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        epoch = 100

        f = equation.initial_condition(mesh.points)
        # u = equation.solution(mesh.points)
        loss_fn = torch.nn.MSELoss()

        losses = []

        with StreamPlotter(filename="poisson.mp4") as plotter:
            plotter.draw_mesh(mesh, f)
            pbar = tqdm(total=epoch)
            for i in range(epoch):
                optimizer.zero_grad()
                K = assembler(mesh.points)
                u = K.solve(f)
                loss = loss_fn(K @ u, f)
                # TODO: why retain_graph=True?
                loss.backward(retain_graph=True) 
                optimizer.step()
                scheduler.step()
                plotter.draw_mesh(mesh, f)
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
                losses.append(loss.item())

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(np.arange(len(losses)), losses, label="loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_yscale("log")
        fig.savefig("loss.png")


.. raw:: html

    <div style="display: flex; justify-content: center; align-items: center;">
    <video width="400" height="400" controls>
      <source src="../_static/poisson_adaptive.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    </div>
    
    
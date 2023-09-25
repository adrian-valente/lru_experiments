from typing import Callable, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import DSModel

def simulate_ds(x0s: torch.Tensor,
                timesteps: int,
                ds: Callable[[torch.Tensor], torch.Tensor],
                deltaT: float = 0.05,
                ) -> torch.Tensor:
    x = x0s
    trajectory = []
    for t in range(timesteps):
        x = x + deltaT * ds(x)
        trajectory.append(x.clone())
    return torch.stack(trajectory, dim=1)


available_ds = {
    "double_well": 1,
    "limit_cycle": 2,
}

def double_well(x: torch.Tensor) -> torch.Tensor:
    return -4 * x**3 + 4 * x


def limit_cycle(x: torch.Tensor) -> torch.Tensor:
    A = torch.Tensor([[0, 1], [-1, 0]])
    return torch.tanh(x @ A)
    

def plot_potential_1d(simulator: Union[DSModel, Callable[[torch.Tensor], torch.Tensor]],
                     limits: Tuple[float, float],
                     ax: plt.Axes = None,
                     plot_kwargs: dict = None,
                     ) -> plt.Axes:
    timesteps = 100
    x0s = torch.linspace(limits[0], limits[1], 100).unsqueeze(1)
    u = torch.zeros((x0s.shape[0], timesteps, 1))
    if isinstance(simulator, DSModel):
        y = simulator.forward(u, x0s).detach().numpy()
    else:    
        y = simulate_ds(x0s, timesteps, simulator).detach().numpy()
    y = y.squeeze(2)
        
    y = np.hstack((x0s.detach().numpy(), y))
    origin, dest = y[:, :-1].ravel(), y[:, 1:].ravel()
    bins_origin = np.linspace(limits[0], limits[1], 80)
    bins_dest = np.zeros(len(bins_origin)-1)
    for i in range(len(bins_origin)-1):
        bins_dest[i] = np.mean(dest[(origin >= bins_origin[i]) & (origin < bins_origin[i+1])])
    bins_origin = (bins_origin[:-1] + bins_origin[1:]) / 2
    V = -np.cumsum(bins_dest - bins_origin)
    
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(bins_origin, V, **plot_kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)")
    return ax
    

if __name__ == '__main__':
    ax = plot_potential_1d(double_well, (-2, 2))
    plt.show()
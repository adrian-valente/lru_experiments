from functools import partial
from typing import Tuple
import torch

import dynamical_systems as dslib

tasks = {
    "flip_flop1": (1, 1),
    "flip_flop2": (2, 2),
    "flip_flop3": (3, 3),
    "double_well": (1, 1),
    "limit_cycle": (2, 2),
}

def flip_flop(d: int,
              timesteps: int = 1000,
              n: int = 5000,
              p: float = 0.2,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate (x, y) data for the flip-flop task.
    """
    x = torch.zeros(n, timesteps, d)
    y = torch.zeros(n, timesteps, d)
    x = torch.bernoulli(torch.ones(n, timesteps, d) * p)
    x = x - 2 * torch.bernoulli(x * 0.5)  # flip half of the bits
    
    cur = torch.zeros(n, d)
    for t in range(timesteps):
        cur = torch.where(x[:, t] != 0, x[:, t], cur)
        y[:, t] = cur
    return x, y

flip_flop1 = partial(flip_flop, d=1)
flip_flop2 = partial(flip_flop, d=2)
flip_flop3 = partial(flip_flop, d=3)

def fit_ds(timesteps: int,
                ds: str,
                n: int = 5000,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ds in dslib.available_ds.keys()
    dims = dslib.available_ds[ds]
    ds_fn = getattr(dslib, ds)
    
    x0s = torch.randn((n, dims))
    trajectories = dslib.simulate_ds(x0s, timesteps, ds_fn)
    u = torch.zeros((n, timesteps, 1))
    if trajectories.isnan().any():
        nanexamples = torch.where(trajectories.isnan().any(dim=1).any(dim=1))[0]
        print(f"Warning: NaNs in {len(nanexamples)} trajectories, removing them")
        trajectories = trajectories[~trajectories.isnan().any(dim=1).any(dim=1)]
    return u, trajectories, x0s


double_well = partial(fit_ds, ds="double_well")
limit_cycle = partial(fit_ds, ds="limit_cycle")
    
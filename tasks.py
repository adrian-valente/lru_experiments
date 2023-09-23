from typing import Tuple
import torch

def flip_flop(d: int,
              timesteps: int,
              n: int,
              p: float = 0.05,
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
    
    
    
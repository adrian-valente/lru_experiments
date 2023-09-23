from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class LRU(nn.Module):
    
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 d_out: int,
                 r_min: float = 0.,
                 r_max: float = 1.,
                 max_phase: float = 6.28
                 ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        
        self.theta_log = nn.Parameter(torch.empty(d_hidden))
        self.nu_log = nn.Parameter(torch.empty(d_hidden))
        self.gamma_log = nn.Parameter(torch.empty(d_hidden))
        self.B_re = nn.Parameter(torch.empty(d_hidden, d_in))
        self.B_im = nn.Parameter(torch.empty(d_hidden, d_in))
        self.C_re = nn.Parameter(torch.empty(d_out, d_hidden))
        self.C_im = nn.Parameter(torch.empty(d_out, d_hidden))
        self.D = nn.Parameter(torch.empty(d_out, d_in))
        
        self._init_params()
    
    def diag_lambda(self) -> torch.Tensor:
        return torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))    
    
    def _init_params(self):

        nn.init.uniform_(self.theta_log, a=0, b=self.max_phase)
        
        u = torch.rand((self.d_hidden,))
        nu_init = torch.log(-0.5 * torch.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        with torch.no_grad():
            self.nu_log.copy_(nu_init)
            diag_lambda = self.diag_lambda()
            self.gamma_log.copy_(torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2)))
        
        nn.init.xavier_normal_(self.B_re)
        nn.init.xavier_normal_(self.B_im)
        nn.init.xavier_normal_(self.C_re)
        nn.init.xavier_normal_(self.C_im)
        nn.init.xavier_normal_(self.D)  # Set something like diagonal matrix eventually
        
    def forward(self, u: torch.Tensor, init_states: torch.Tensor = None) -> torch.Tensor:
        diag_lambda = self.diag_lambda()
        B_norm = torch.diag(self.gamma_log).to(torch.cfloat) @ (self.B_re + 1j * self.B_im)
        C = self.C_re + 1j * self.C_im
        
        # Initial states can be a vector of shape (d_hidden,) or a matrix of shape (batch_size, d_hidden)
        if init_states is not None and init_states.ndim == 1:
            init_states = init_states.unsqueeze(0)
        
        h = init_states.to(torch.cfloat) if init_states is not None \
                else torch.zeros((u.shape[0], self.d_hidden), dtype=torch.cfloat)
        outputs = []
        for t in range(u.shape[1]):
            h = h * diag_lambda + u[:, t].to(torch.cfloat) @ B_norm.T
            y = torch.real(h @ C.T) + u[:, t] @ self.D.T
            outputs.append(y)
        return torch.stack(outputs, dim=1)
    

class SequenceLayer(nn.Module):
    
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 layer_widths: List[int],
                 non_linearity: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 skip_connection: bool = False
                 ) -> None:
        super().__init__()
        self.layer_widths = layer_widths
        self.non_linearity = non_linearity
        self.skip_connection = skip_connection
        
        self.LRU = LRU(d_in, d_hidden, layer_widths[0])
        self.layers = nn.ModuleList([nn.Linear(layer_widths[i], layer_widths[i+1]) for i in range(len(layer_widths)-1)])
        
    def forward(self, u: torch.Tensor, init_states: torch.Tensor = None) -> torch.Tensor:
        y = self.LRU(u, init_states)
        for layer in self.layers[:-1]:
            y = layer(y)
            y = self.non_linearity(y)
        y = self.layers[-1](y)
        if self.skip_connection:
            y = y + u
        return y
    
    
class DeepLRUModel(nn.Module):
    
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 depth: int,
                 internal_widths: List[int],
                 output_widths: List[int] = None,
                 non_linearity: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 skip_connection: bool = False
                 ) -> None:
        super().__init__()
        self.depth = depth
        
        if output_widths is None:
            output_widths = internal_widths
        
        layers = []
        for i in range(depth-1):
            layers.append(SequenceLayer(d_in, d_hidden, internal_widths, non_linearity, skip_connection))
        layers.append(SequenceLayer(d_in, d_hidden, output_widths, non_linearity, skip_connection))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, u: torch.Tensor, init_states: List[torch.Tensor] = None) -> torch.Tensor:
        y = u
        if init_states is not None:
            assert len(init_states) == self.depth
        for i, layer in enumerate(self.layers):
            y = layer(y, init_states[i] if init_states is not None else None)
        return y
        
        
class ModelDS(nn.Module):
    
    def __init__(self,
                 d_ds: int,
                 d_in: int,
                 d_hidden: int,
                 depth: int,
                 internal_widths: List[int],
                 encoder_widths: List[int],
                 encoder_non_linearity: Callable[[torch.Tensor], torch.Tensor] = F.relu,
                 output_widths: List[int] = None,
                 lru_kwargs: dict = None
                 ) -> None:
        super().__init__()
        self.d_ds = d_ds
        self.d_in = d_in
        self.encoder_non_linearity = encoder_non_linearity
        
        encoder_layers = []
        encoder_widths.insert(0, d_ds)
        assert encoder_widths[-1] == (d_hidden * depth), "Encoder output width must be equal to d_hidden * depth"
        for i in range(len(encoder_widths)-1):
            encoder_layers.append(nn.Linear(encoder_widths[i], encoder_widths[i+1]))
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.lru_model = DeepLRUModel(d_in, d_hidden, depth, internal_widths, output_widths, **lru_kwargs)
        
    def forward(self, u: torch.Tensor, init_states: List[torch.Tensor] = None) -> torch.Tensor:
        
        

if __name__ == '__main__':
    net = LRU(1, 4, 1)
    inputs = torch.Tensor([[0., 0, 1, 1, 0]]).unsqueeze(2)
    net(inputs)
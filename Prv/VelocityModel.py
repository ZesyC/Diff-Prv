import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class VelocityModel(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5, cond_dim=0):
        super(VelocityModel, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm
        self.cond_dim = cond_dim

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        self.cond_emb_layer = nn.Linear(cond_dim, self.time_emb_dim) if cond_dim > 0 else None

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        film_dims = in_dims_temp[1:] + out_dims_temp[1:]
        self.film_layers = nn.ModuleList([nn.Linear(self.time_emb_dim, dim * 2) for dim in film_dims]) if cond_dim > 0 else None

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

        if self.cond_emb_layer is not None:
            size = self.cond_emb_layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            self.cond_emb_layer.weight.data.normal_(0.0, std)
            self.cond_emb_layer.bias.data.zero_()

        if self.film_layers is not None:
            for layer in self.film_layers:
                layer.weight.data.zero_()
                layer.bias.data.zero_()

    def _apply_film(self, h, cond_emb, layer_idx):
        if self.film_layers is None:
            return h
        gamma, beta = self.film_layers[layer_idx](cond_emb).chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta

    def forward(self, x, t, condition=None, mess_dropout=True):
        device = x.device
        
        # Scale t from [0, 1] to [0, 1000] for better embedding distinction if desired, 
        # but using t directly is also fine. We will scale by 1000.
        timesteps = t * 1000.0
        
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
            
        emb = self.emb_layer(time_emb)
        cond_emb = None
        if self.cond_emb_layer is not None:
            if condition is None:
                condition = torch.zeros(x.size(0), self.cond_dim, device=device, dtype=x.dtype)
            cond_emb = self.cond_emb_layer(condition)
            emb = emb + cond_emb
        
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x)
            
        h = torch.cat([x, emb], dim=-1)
        
        film_idx = 0
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = self._apply_film(h, cond_emb, film_idx)
            film_idx += 1
            h = torch.tanh(h)
            
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            h = self._apply_film(h, cond_emb, film_idx)
            film_idx += 1
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

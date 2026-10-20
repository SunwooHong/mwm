"""Optional gradient-free residual on TD-MPC2 latent dynamics. Not used unless attached."""

import torch
import torch.nn.functional as F


def simnorm(x, dim):
    shp = x.shape
    x = F.softmax(x.reshape(*shp[:-1], -1, dim), dim=-1)
    return x.reshape(*shp)


def inv_simnorm(x, dim):
    shp = x.shape
    x = x.reshape(*shp[:-1], -1, dim).clamp(min=1e-8)
    return torch.log(x).reshape(*shp)


class HebbianLatentResidual:
    def __init__(self, latent_dim, action_dim, simnorm_dim, feature_dim=64, eta=0.05, decay=0.001, max_norm=10.0, seed=0):
        g = torch.Generator()
        g.manual_seed(seed)
        proj = torch.randn(latent_dim + action_dim, feature_dim, generator=g)
        proj = proj / proj.norm(dim=0, keepdim=True).clamp(min=1e-6)
        self.proj = proj
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.simnorm_dim = simnorm_dim
        self.feature_dim = feature_dim
        self.eta = eta
        self.decay = decay
        self.max_norm = max_norm
        self.eps = 1e-6
        self.read = True
        self.write = True
        self.P = torch.zeros(latent_dim, feature_dim)
        self.proj_hits = 0
        self.steps = 0

    def to(self, device):
        self.proj = self.proj.to(device)
        self.P = self.P.to(device)
        return self

    def phi(self, z, a):
        raw = torch.cat([z, a], dim=-1)
        raw = raw / raw.norm(dim=-1, keepdim=True).clamp(min=1.0)
        return raw @ self.proj.to(dtype=raw.dtype, device=raw.device)

    def correct(self, z, a, z_hat):
        if not self.read:
            return z_hat
        r = self.phi(z, a) @ self.P.to(device=z.device, dtype=z.dtype).T
        return simnorm(inv_simnorm(z_hat, self.simnorm_dim) + r, self.simnorm_dim)

    @torch.no_grad()
    def update(self, z, a, z_pred, z_obs):
        """Score happens before this call. z_pred is the prediction being supervised."""
        self.steps += 1
        if not self.write:
            return
        err = (z_obs - z_pred).detach()
        x = self.phi(z, a).detach()
        scale = 1.0 / (x.pow(2).sum(-1, keepdim=True) + self.eps)
        P = (1.0 - self.decay) * self.P + self.eta * (err * scale).T @ x
        flat = torch.linalg.vector_norm(P).clamp(min=1e-12)
        if float(flat) > self.max_norm:
            self.proj_hits += 1
            P = P * (self.max_norm / flat)
        self.P = P

    def memory_bytes(self):
        return self.P.numel() * 4

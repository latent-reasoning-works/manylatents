"""ScoreDiffusionModule — a small VE denoising-score-matching model (reusable, guarded torch).

The point is the intrinsic-dimension COMPARISON (data-side LID vs model-side score-ID, Stanczuk et
al. ICML 2024), not a SOTA generator: a 3-4 layer MLP score net trains in minutes on 50-D PCA data.

Parameterisation (noise-prediction / EDM-lite, VE forward x_t = x0 + sigma*eps):
  net predicts eps_hat(x_t, sigma);  score(x_t, sigma) = -eps_hat / sigma  (= grad_x log p_sigma);
  denoise (Tweedie) E[x0 | x_t] = x_t + sigma^2 * score = x_t - sigma * eps_hat.
Training minimises E || eps_hat(x0 + sigma*eps, sigma) - eps ||^2 over log-uniform sigma.

Everything is computed in a standardised space (per-dim mean/std stored at fit) — dimension counts
from the score spectrum are scale-invariant, and standardisation stabilises training.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("ScoreDiffusionModule requires torch (optional dep). `uv sync` the "
                          "manylatents env, or install torch.") from e


class ScoreDiffusionModule:
    """Denoising score-matching MLP. `.fit(X)`, `.score(x, sigma)`, `.denoise(x, sigma)`, `.sample(n)`."""

    def __init__(self, hidden: int = 256, depth: int = 4, n_fourier: int = 16,
                 sigma_min: float | None = None, sigma_max: float | None = None,
                 lr: float = 1e-3, epochs: int = 200, batch_size: int = 256,
                 device: str | None = None, seed: int = 42):
        _require_torch()
        self.hidden, self.depth, self.n_fourier = hidden, depth, n_fourier
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.lr, self.epochs, self.batch_size, self.seed = lr, epochs, batch_size, seed
        from manylatents.utils.backend import resolve_device
        self.device = resolve_device(device)
        self.net = None
        self._mean = self._std = None
        self._is_fitted = False

    # ---- net ----
    def _build_net(self, d):
        import torch
        import torch.nn as nn

        n_fourier = self.n_fourier

        class ScoreNet(nn.Module):
            def __init__(s):
                super().__init__()
                s.register_buffer("freqs", torch.randn(n_fourier) * 2.0)  # fixed Fourier freqs for log-sigma
                s.inp = nn.Linear(d + 2 * n_fourier, self.hidden)
                s.blocks = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(self.depth)])
                s.out = nn.Linear(self.hidden, d)
                s.act = nn.SiLU()

            def forward(s, x, log_sigma):
                ls = log_sigma.reshape(-1, 1) * s.freqs.reshape(1, -1)
                semb = torch.cat([torch.sin(ls), torch.cos(ls)], dim=1)
                h = s.act(s.inp(torch.cat([x, semb], dim=1)))
                for blk in s.blocks:
                    h = h + s.act(blk(h))   # residual
                return s.out(h)

        return ScoreNet().to(self.device)

    def _standardize(self, X):
        return (X - self._mean) / self._std

    def _unstandardize(self, Z):
        return Z * self._std + self._mean

    # ---- fit ----
    def fit(self, X, y=None):
        import torch
        X = np.asarray(X, np.float32)
        n, d = X.shape
        self._mean = X.mean(0, keepdims=True)
        self._std = X.std(0, keepdims=True) + 1e-6
        Z = self._standardize(X)
        if self.sigma_max is None:
            self.sigma_max = 5.0                    # standardised data has unit scale
        if self.sigma_min is None:
            self.sigma_min = 0.01
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = self._build_net(d)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Zt = torch.tensor(Z, device=self.device)
        log_lo, log_hi = np.log(self.sigma_min), np.log(self.sigma_max)
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                x0 = Zt[idx]
                log_sigma = torch.rand(x0.shape[0], device=self.device) * (log_hi - log_lo) + log_lo
                sigma = log_sigma.exp().reshape(-1, 1)
                eps = torch.randn_like(x0)
                x_t = x0 + sigma * eps
                eps_hat = self.net(x_t, log_sigma)
                loss = ((eps_hat - eps) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * x0.shape[0]
            if ep % max(1, self.epochs // 5) == 0 or ep == self.epochs - 1:
                logger.info(f"ScoreDiffusionModule fit: epoch {ep} DSM loss={tot / n:.4f}")
        self.net.eval()
        self._is_fitted = True
        return self

    # ---- inference (standardised space) ----
    def _eps_hat(self, Z, sigma):
        import torch
        Z = np.asarray(Z, np.float32)
        sig = np.full((Z.shape[0],), float(sigma), np.float32) if np.isscalar(sigma) else np.asarray(sigma, np.float32)
        with torch.no_grad():
            zt = torch.tensor(Z, device=self.device)
            ls = torch.log(torch.tensor(sig, device=self.device))
            return self.net(zt, ls).cpu().numpy()

    def score(self, x, sigma, standardized=True):
        """grad_x log p_sigma. If standardized=False, x is in raw space (standardised internally)."""
        Z = np.asarray(x, np.float32) if standardized else self._standardize(np.asarray(x, np.float32))
        return -self._eps_hat(Z, sigma) / float(sigma)

    def denoise(self, x, sigma, standardized=True):
        Z = np.asarray(x, np.float32) if standardized else self._standardize(np.asarray(x, np.float32))
        Z0 = Z - float(sigma) * self._eps_hat(Z, sigma)     # Tweedie
        return Z0 if standardized else self._unstandardize(Z0)

    def sample(self, n, steps=200, standardized=False, seed=None):
        """Annealed Langevin over the log-spaced noise ladder (basic; quality proxy only)."""
        import torch
        rng = np.random.default_rng(self.seed if seed is None else seed)
        d = self._mean.shape[1]
        Z = rng.normal(0, self.sigma_max, size=(n, d)).astype(np.float32)
        sigmas = np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min), steps)).astype(np.float32)
        for sigma in sigmas:
            s = self.score(Z, sigma)
            step = 0.5 * (sigma ** 2)
            Z = Z + step * s + np.sqrt(2 * step) * rng.normal(size=Z.shape).astype(np.float32)
        return Z if standardized else self._unstandardize(Z)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
class PRLoss(nn.Module):
    """Participation‐Ratio penalty:  λ_pr * PR(z)."""
    def __init__(self, lambda_pr: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_pr = lambda_pr
        self.eps       = eps

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, d]
        B, d = latent.shape
        zc   = latent - latent.mean(0, keepdim=True)
        cov  = (zc.T @ zc) / (B - 1 + self.eps)
        eig  = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        pr   = (eig.sum() ** 2) / eig.pow(2).sum()
        return self.lambda_pr * pr


# -----------------------------------------------------------------------------
class AnisotropyLoss(nn.Module):
    """Anisotropy penalty:  λ_aniso * (1 – anisotropy(z))."""
    def __init__(self, lambda_aniso: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_aniso = lambda_aniso
        self.eps          = eps

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, d = latent.shape
        zc   = latent - latent.mean(0, keepdim=True)
        cov  = (zc.T @ zc) / (B - 1 + self.eps)
        eig  = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        aniso = eig.max() / eig.sum()
        return self.lambda_aniso * (1.0 - aniso)


# -----------------------------------------------------------------------------
class TSALoss(nn.Module):
    """
    Tangent‐Space Approximation: λ_tsa * E[‖P_z – P_x‖₁²] over each minibatch neighbor set.
    We compute raw‐space PCA on CPU to save GPU memory.
    """
    def __init__(self, lambda_tsa: float = 0.1, k: int = 25, p: int = 1, eps: float = 1e-8):
        super().__init__()
        self.lambda_tsa = lambda_tsa
        self.k          = k
        self.p          = p
        self.eps        = eps

    def forward(self, latent: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # latent: [B, d], raw: [B, D]
        B, D = raw.shape

        # pairwise distances on GPU
        dists = torch.cdist(raw, raw)
        nbrs  = dists.topk(self.k + 1, largest=False).indices[:, 1:]  # [B, k]

        losses = []
        for i in range(B):
            nn_idx = nbrs[i]                # LongTensor[k]
            Zn     = latent[nn_idx]         # [k, d]
            Xn     = raw   [nn_idx]         # [k, D]

            # — latent‐space projector —
            Zc = Zn - Zn.mean(0, keepdim=True)
            Cz = (Zc.T @ Zc) / (self.k - 1 + self.eps)
            eig_z, Uz = torch.linalg.eigh(Cz)
            Ui = Uz[:, -self.p:]             # [d, p]
            Pz = Ui @ Ui.T                   # [d, d]

            # — raw‐space projector on CPU —
            Xn_cpu = Xn.detach().cpu().numpy()             # [k, D]
            Xc_cpu = Xn_cpu - Xn_cpu.mean(axis=0, keepdims=True)
            Cx_cpu = (Xc_cpu.T @ Xc_cpu) / (self.k - 1 + self.eps)  # [D, D]
            # numpy eigh on CPU
            eig_x, Ux = np.linalg.eigh(Cx_cpu)
            Vi = torch.from_numpy(Ux[:, -self.p:]).to(Pz.device)   # [D, p]
            Px = Vi @ Vi.T                                         # [D, D] on GPU

            # Frobenius‐norm squared
            losses.append(torch.norm(Pz - Px) ** 2)

        tsa_val = torch.stack(losses).mean()
        return self.lambda_tsa * tsa_val


# -----------------------------------------------------------------------------
class AllGeomLoss(nn.Module):
    """
    Composite: MSE + PR + Anisotropy + TSA, but *only* sums each penalty once.
    Exposes a .components(...) method so you can log/ablate each term.
    """
    def __init__(
        self,
        lambda_pr:    float = 0.01,
        lambda_aniso: float = 0.01,
        lambda_tsa:   float = 0.1,
        k:            int   = 25,
        p:            int   = 1,
    ):
        super().__init__()
        self.mse    = F.mse_loss
        self.pr     = PRLoss(lambda_pr=lambda_pr)
        self.aniso  = AnisotropyLoss(lambda_aniso=lambda_aniso)
        self.tsa    = TSALoss(lambda_tsa=lambda_tsa, k=k, p=p)

    def components(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        latent:  torch.Tensor,
        raw:     torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        recon = self.mse(outputs, targets)
        pr    = self.pr(latent)
        aniso = self.aniso(latent)
        tsa   = self.tsa(latent, raw)
        return {"recon": recon, "pr": pr, "aniso": aniso, "tsa": tsa}

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        latent:  torch.Tensor,
        raw:     torch.Tensor,
    ) -> torch.Tensor:
        comps = self.components(outputs, targets, latent, raw)
        return sum(comps.values())

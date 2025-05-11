import torch
import torch.nn as nn
import torch.nn.functional as F


class PRLoss(nn.Module):
    """
    Loss wrapper for Participation Ratio:
    Computes MSE + lambda_pr * PR(z)
    PR = (sum eig)^2 / sum(eig^2), eig from batch covariance of latent z.
    """
    def __init__(self, lambda_pr: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_pr = lambda_pr
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, latent: torch.Tensor = None, **_):
        recon = F.mse_loss(outputs, targets)
        # Participation Ratio
        B, d = latent.shape
        zc = latent - latent.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1 + self.eps)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        pr_val = (eig.sum() ** 2) / (eig.pow(2).sum())
        return recon + self.lambda_pr * pr_val

class AnisotropyLoss(nn.Module):
    """
    Loss wrapper for Anisotropy:
    Computes MSE + lambda_aniso * (1 - anisotropy(z))
    anisotropy = 1 - max(eig)/sum(eig)
    """
    def __init__(self, lambda_aniso: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_aniso = lambda_aniso
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, latent: torch.Tensor = None, **_):
        recon = F.mse_loss(outputs, targets)
        B, d = latent.shape
        zc = latent - latent.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1 + self.eps)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        aniso_val = eig.max() / eig.sum()
        return recon + self.lambda_aniso * (1.0 - aniso_val)

class TSALoss(nn.Module):
    """
    Loss wrapper for Tangent-Space Approximation:
    Computes MSE + lambda_tsa * TSA(raw, z)
    TSA = average Frobenius norm squared between raw- and latent-space tangent projectors
    """
    def __init__(self, lambda_tsa: float = 0.1, k: int = 25, p: int = 1, eps: float = 1e-8):
        super().__init__()
        self.lambda_tsa = lambda_tsa
        self.k = k
        self.p = p
        self.eps = eps

    def forward(self,
                outputs: torch.Tensor,
                targets: torch.Tensor,
                latent: torch.Tensor = None,
                raw: torch.Tensor = None,
                **_):
        recon = F.mse_loss(outputs, targets)
        B, D = raw.shape
        _, d = latent.shape
        # pairwise distances in raw space
        dists = torch.cdist(raw, raw)
        idx = dists.topk(self.k + 1, largest=False).indices
        losses = []
        for i in range(B):
            nbrs = idx[i, 1:]
            Z = latent[nbrs]
            X = raw[nbrs]
            Zc = Z - Z.mean(0, keepdim=True)
            Xc = X - X.mean(0, keepdim=True)
            Cz = (Zc.T @ Zc) / (self.k - 1 + self.eps)
            Cx = (Xc.T @ Xc) / (self.k - 1 + self.eps)
            _, Uz = torch.linalg.eigh(Cz)
            Ui = Uz[:, -self.p:]
            _, Ux = torch.linalg.eigh(Cx)
            Vi = Ux[:, -self.p:]
            Pz = Ui @ Ui.T
            Px = Vi @ Vi.T
            losses.append(torch.norm(Pz - Px) ** 2)
        tsa_val = torch.stack(losses).mean()
        return recon + self.lambda_tsa * tsa_val

class AllGeomLoss(nn.Module):
    """
    Composite loss: sum of PR, Anisotropy, and TSA sub-losses
    """
    def __init__(self,
                 loss_pr:    nn.Module,
                 loss_aniso: nn.Module,
                 loss_tsa:   nn.Module):
        super().__init__()
        self.pr_loss    = loss_pr
        self.aniso_loss = loss_aniso
        self.tsa_loss   = loss_tsa

    def forward(self, outputs, targets, latent=None, raw=None, **_):
        recon  = F.mse_loss(outputs, targets)
        pr     = self.pr_loss(outputs, targets, latent=latent)
        aniso  = self.aniso_loss(outputs, targets, latent=latent)
        tsa    = self.tsa_loss(outputs, targets, latent=latent, raw=raw)
        return recon + pr + aniso + tsa
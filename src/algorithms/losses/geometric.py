# src/algorithms/losses/geometric.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRLoss(nn.Module):
    def __init__(self, lambda_pr: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_pr = lambda_pr
        self.eps = eps

    def components(self, outputs, targets, latent=None, **_):
        recon = F.mse_loss(outputs, targets)
        B, d = latent.shape
        zc = latent - latent.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1 + self.eps)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        pr_val = (eig.sum() ** 2) / (eig.pow(2).sum())
        return {
            "recon": recon,
            "pr": self.lambda_pr * pr_val
        }

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["pr"]


class AnisotropyLoss(nn.Module):
    def __init__(self, lambda_aniso: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_aniso = lambda_aniso
        self.eps = eps

    def components(self, outputs, targets, latent=None, **_):
        recon = F.mse_loss(outputs, targets)
        B, d = latent.shape
        zc = latent - latent.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1 + self.eps)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        aniso_val = eig.max() / eig.sum()
        return {
            "recon": recon,
            "aniso": self.lambda_aniso * (1.0 - aniso_val)
        }

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["aniso"]


class TSALoss(nn.Module):
    def __init__(self, lambda_tsa: float = 0.1, k: int = 25, p: int = 1, eps: float = 1e-8):
        super().__init__()
        self.lambda_tsa = lambda_tsa
        self.k = k
        self.p = p
        self.eps = eps

    def components(self, outputs, targets, latent=None, raw=None, **_):
        recon = F.mse_loss(outputs, targets)
        B, _ = raw.shape

        # build a kNN on this minibatch
        dists = torch.cdist(raw, raw)
        idx   = dists.topk(self.k + 1, largest=False).indices  # [B, k+1]
        tsa_losses = []
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
            tsa_losses.append(torch.norm(Pz - Px) ** 2)
        tsa_val = torch.stack(tsa_losses).mean()

        return {
            "recon": recon,
            "tsa": self.lambda_tsa * tsa_val
        }

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["tsa"]


class AllGeomLoss(nn.Module):
    """
    Composite: sum of PR, Anisotropy, and TSA.
    """
    def __init__(self,
                 loss_pr:    PRLoss,
                 loss_aniso: AnisotropyLoss,
                 loss_tsa:   TSALoss):
        super().__init__()
        self.pr_loss    = loss_pr
        self.aniso_loss = loss_aniso
        self.tsa_loss   = loss_tsa

    def components(self, outputs, targets, latent=None, raw=None, **_):
        c1 = self.pr_loss.components(outputs, targets, latent=latent)
        c2 = self.aniso_loss.components(outputs, targets, latent=latent)
        c3 = self.tsa_loss.components(outputs, targets, latent=latent, raw=raw)
        # they all include 'recon' so we pick one
        recon = c1["recon"]
        return {
            "recon": recon,
            "pr":    c1["pr"],
            "aniso": c2["aniso"],
            "tsa":   c3["tsa"]
        }

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["pr"] + comps["aniso"] + comps["tsa"]

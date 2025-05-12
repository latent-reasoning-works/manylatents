import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    """
    Helper to project high-dimensional "raw" inputs into a lower-dimensional space
    before geometry-based loss computations.
    """
    def __init__(self, input_dim: int, proj_dim: int = 512, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PRLoss(nn.Module):
    def __init__(self, lambda_pr: float = 0.01, eps: float = 1e-8):
        super().__init__()
        self.lambda_pr = lambda_pr
        self.eps = eps

    def components(self, outputs, targets, latent=None, **_):
        recon = F.mse_loss(outputs, targets)
        if self.lambda_pr == 0: # Skip heavy compute if weight is zero
            return {"recon": recon, "pr": recon.new_tensor(0.)}
        B, d = latent.shape
        zc = latent - latent.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1 + self.eps)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        pr_val = (eig.sum() ** 2) / (eig.pow(2).sum())
        return {"recon": recon, "pr": self.lambda_pr * pr_val}

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
        if self.lambda_aniso == 0:  # Skip heavy compute if weight is zero
            return {"recon": recon, "aniso": recon.new_tensor(0.)}
        B, d = latent.shape
        zc = latent - latent.mean(0, keepdim=True)
        cov = (zc.T @ zc) / (B - 1 + self.eps)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        aniso_val = eig.max() / eig.sum()
        return {"recon": recon, "aniso": self.lambda_aniso * (1.0 - aniso_val)}

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["aniso"]


class TSALoss(nn.Module):
    def __init__(
        self,
        lambda_tsa: float = 0.1,
        k: int = 25,
        p: int = 1,
        eps: float = 1e-8,
        raw_proj: bool = False,
        input_dim: int = None,
        proj_dim: int = 512,
        proj_bias: bool = False,
        offload_cpu: bool = True,
    ):
        super().__init__()
        self.lambda_tsa = lambda_tsa
        self.k = k
        self.p = p
        self.eps = eps
        self.offload_cpu = offload_cpu
        if raw_proj:
            assert input_dim is not None, "input_dim must be provided if raw_proj is True"
            self.raw_proj = LinearProjection(input_dim=input_dim, proj_dim=proj_dim, bias=proj_bias)
        else:
            self.raw_proj = None

    def components(self, outputs, targets, latent=None, raw=None, **_):
        recon = F.mse_loss(outputs, targets)
        if self.lambda_tsa == 0: # skip heavy compute if weight is zero
            return {"recon": recon, "tsa": recon.new_tensor(0.)}
        if self.raw_proj is not None:
            raw = self.raw_proj(raw)
        if self.offload_cpu:
            raw = raw.detach().cpu()
            latent = latent.detach().cpu()
        B, D = raw.shape
        dists = torch.cdist(raw, raw) # pairwise distances in projected space
        idx = dists.topk(self.k + 1, largest=False).indices
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
            _, Ux = torch.linalg.eigh(Cx)
            Ui = Uz[:, -self.p:]
            Vi = Ux[:, -self.p:]
            Pz = Ui @ Ui.T
            Px = Vi @ Vi.T
            tsa_losses.append(torch.norm(Pz - Px) ** 2)
        tsa_val = torch.stack(tsa_losses).mean()
        tsa_term = self.lambda_tsa * tsa_val
        # Ensure TSA term is on same device as outputs
        tsa_term = tsa_term.to(outputs.device)
        return {"recon": recon, "tsa": tsa_term}

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["tsa"]


class AllGeomLoss(nn.Module):
    """
    Composite: sum of PR, Anisotropy, and TSA.
    """
    def __init__(self, loss_pr: PRLoss, loss_aniso: AnisotropyLoss, loss_tsa: TSALoss):
        super().__init__()
        self.pr_loss = loss_pr
        self.aniso_loss = loss_aniso
        self.tsa_loss = loss_tsa

    def components(self, outputs, targets, latent=None, raw=None, **_):
        # Compute recon once
        recon = F.mse_loss(outputs, targets)
        # PR component
        pr = self.pr_loss.components(outputs, targets, latent=latent)["pr"]
        # Anisotropy component
        aniso = self.aniso_loss.components(outputs, targets, latent=latent)["aniso"]
        # TSA component
        tsa = self.tsa_loss.components(outputs, targets, latent=latent, raw=raw)["tsa"]
        return {"recon": recon, "pr": pr, "aniso": aniso, "tsa": tsa}

    def forward(self, outputs, targets, **extras):
        comps = self.components(outputs, targets, **extras)
        return comps["recon"] + comps["pr"] + comps["aniso"] + comps["tsa"]

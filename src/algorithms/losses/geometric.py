import hydra
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


class AEDimLoss(nn.Module):
    def __init__(
        self,
        lambda_tsa: float,
        lambda_pr: float,
        tsa_cfg:   DictConfig,
        pr_cfg:    DictConfig,
    ):
        super().__init__()
        self.lambda_tsa = lambda_tsa
        self.lambda_pr  = lambda_pr
        self.tsa = hydra.utils.instantiate(tsa_cfg)
        self.pr  = hydra.utils.instantiate(pr_cfg)

    def forward(self, outputs, targets, latent=None, **_):
        # recon
        recon = F.mse_loss(outputs, targets)
        # metric terms
        tsa_loss = self.tsa(embeddings=latent, dataset=None, module=None)
        pr_loss  = self.pr(embeddings=latent, dataset=None, module=None)
        return recon + self.lambda_tsa * tsa_loss + self.lambda_pr * pr_loss


class AENeighborhoodLoss(nn.Module):
    def __init__(
        self,
        lambda_trust: float,
        lambda_cont:  float,
        trust_cfg:    DictConfig,
        cont_cfg:     DictConfig,
    ):
        super().__init__()
        self.lambda_trust = lambda_trust
        self.lambda_cont  = lambda_cont
        self.trust = hydra.utils.instantiate(trust_cfg)
        self.cont  = hydra.utils.instantiate(cont_cfg)

    def forward(self, outputs, targets, latent=None, dataset=None, **_):
        recon = F.mse_loss(outputs, targets)
        t = 1.0 - self.trust(embeddings=latent, dataset=dataset, module=None)
        c = 1.0 - self.cont(embeddings=latent, dataset=dataset, module=None)
        return recon + self.lambda_trust * t + self.lambda_cont * c


class AEShapeLoss(nn.Module):
    def __init__(
        self,
        lambda_pr:    float,
        lambda_aniso: float,
        pr_cfg:       DictConfig,
        aniso_cfg:    DictConfig,
    ):
        super().__init__()
        self.lambda_pr    = lambda_pr
        self.lambda_aniso = lambda_aniso
        self.pr          = hydra.utils.instantiate(pr_cfg)
        self.aniso       = hydra.utils.instantiate(aniso_cfg)

    def forward(self, outputs, targets, latent=None, dataset=None, **_):
        recon = F.mse_loss(outputs, targets)
        pr_l  = self.pr(embeddings=latent, dataset=dataset, module=None)
        a_l   = 1.0 - self.aniso(embeddings=latent, dataset=dataset, module=None)
        return recon + self.lambda_pr * pr_l + self.lambda_aniso * a_l

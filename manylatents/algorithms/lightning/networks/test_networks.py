## currently not confirmed to work
import pytest
import torch

from manylatents.algorithms.networks.aanet import VAE, Vanilla
from manylatents.algorithms.networks.autoencoder import Autoencoder

# call styles:
#  - "kw"  : AE.loss_function(outputs=…, targets=…)
#  - "one" : Vanilla.loss_function(outputs_tuple)
#  - "pos" : VAE.loss_function(*outputs_list)
NETS = [
    (Autoencoder, dict(input_dim=20, hidden_dims=[16, 8], latent_dim=4), "kw"),
    (Vanilla,     dict(input_dim=20, n_archetypes=4, device=torch.device("cpu")), "one"),
    (VAE,         dict(input_dim=20, n_archetypes=5, device=torch.device("cpu")), "pos"),
]

@pytest.mark.parametrize("net_cls, init_kwargs, call_style", NETS)
def test_forward_exists(net_cls, init_kwargs, call_style):
    batch = torch.randn(3, init_kwargs["input_dim"])
    net = net_cls(**init_kwargs)  # AE stays on CPU by default; AAnet on cpu via device
    outputs = net(batch)          # must run without error
    # final checks
    assert isinstance(outputs, torch.Tensor)
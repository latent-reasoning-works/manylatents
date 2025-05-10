# src/algorithms/networks/test_networks.py

import pytest
import torch

from src.algorithms.networks.aanet import VAE, Vanilla
from src.algorithms.networks.autoencoder import Autoencoder

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
def test_forward_and_loss_exist_and_run(net_cls, init_kwargs, call_style):
    batch = torch.randn(3, init_kwargs["input_dim"])
    net = net_cls(**init_kwargs)  # AE stays on CPU by default; AAnet on cpu via device
    outputs = net(batch)          # must run without error
    
    # call the appropriate loss signature
    if call_style == "kw":
        # Autoencoder: expects keywords
        loss_ret = net.loss_function(outputs=outputs, targets=batch)
    elif call_style == "one":
        # Vanilla: expects a single tuple/list
        assert isinstance(outputs, (tuple, list))
        loss_ret = net.loss_function(outputs)
    else:  # "pos"
        # VAE: expects each element as separate positional arg
        assert isinstance(outputs, (tuple, list))
        loss_ret = net.loss_function(*outputs)
    
    # normalize dict vs tensor return
    if isinstance(loss_ret, dict):
        assert "loss" in loss_ret
        loss = loss_ret["loss"]
    else:
        loss = loss_ret

    # final checks
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

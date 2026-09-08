"""Decoder architecture compatibility and optional asymmetry."""

import pytest
import torch
from torch import nn

from manylatents.algorithms.lightning.networks.autoencoder import Autoencoder
from manylatents.algorithms.lightning.networks.network import HasDecode


@pytest.mark.parametrize("decoder_hidden_dims", [None, [9, 7, 5], []])
def test_decoder_keyword_only_preserves_positional_init_seed(decoder_hidden_dims):
    import inspect

    signature = inspect.signature(Autoencoder)
    assert signature.parameters["decoder_hidden_dims"].kind is inspect.Parameter.KEYWORD_ONLY
    first = Autoencoder(4, [8, 6], 2, "tanh", False, 0.0, 123,
                        decoder_hidden_dims=decoder_hidden_dims)
    torch.manual_seed(999)
    second = Autoencoder(4, [8, 6], 2, "tanh", False, 0.0, init_seed=123,
                         decoder_hidden_dims=decoder_hidden_dims)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("hidden_dims", [[8, 6], 7, []])
@pytest.mark.parametrize("batchnorm,dropout", [(False, 0.0), (True, 0.2)])
def test_default_matches_legacy_architecture_and_weights(hidden_dims, batchnorm, dropout):
    # Freeze the original construction order, including encoder RNG consumption.
    widths = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
    torch.manual_seed(12)
    legacy = nn.Module()
    act = nn.Tanh()
    for name, start, hidden, end in (
        ("encoder", 4, widths, 2),
        ("decoder", 2, reversed(widths), 4),
    ):
        layers = []
        prev = start
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, end))
        setattr(legacy, name, nn.Sequential(*layers))

    torch.manual_seed(12)
    model = Autoencoder(4, hidden_dims, 2, "tanh", batchnorm, dropout)
    assert repr(model.encoder) == repr(legacy.encoder)
    assert repr(model.decoder) == repr(legacy.decoder)
    assert model.state_dict().keys() == legacy.state_dict().keys()
    for key, value in legacy.state_dict().items():
        assert torch.equal(model.state_dict()[key], value)
    model.load_state_dict(legacy.state_dict(), strict=True)
    model.eval()
    legacy.eval()
    x = torch.randn(5, 4)
    expected = legacy.decoder(legacy.encoder(x))
    assert torch.equal(model(x), expected)
    assert torch.equal(model.decode(model.encode(x)), expected)
    assert isinstance(model, HasDecode)


@pytest.mark.parametrize("decoder_hidden_dims", [[9, 7, 5], 9, []])
def test_asymmetric_decoder_round_trip(decoder_hidden_dims):
    model = Autoencoder(4, [8, 6], 2, decoder_hidden_dims=decoder_hidden_dims)
    hidden = [decoder_hidden_dims] if isinstance(decoder_hidden_dims, int) else decoder_hidden_dims
    decoder_dims = [2, *hidden, 4]
    assert [(layer.in_features, layer.out_features) for layer in model.decoder
            if isinstance(layer, nn.Linear)] == list(zip(decoder_dims, decoder_dims[1:]))
    x = torch.randn(5, 4)
    z = model.encode(x)
    decoded = model.decode(z)
    assert z.shape == (5, 2)
    assert decoded.shape == x.shape
    reconstruction, latent = model(x, return_latent=True)
    assert torch.equal(reconstruction, decoded)
    assert torch.equal(latent, z)


@pytest.mark.parametrize("dims", [[0], [-1], [1.5], [True], 3.5])
def test_invalid_decoder_widths_are_rejected(dims):
    with pytest.raises(ValueError, match="decoder_hidden_dims.*positive integer"):
        Autoencoder(4, [8, 6], 2, decoder_hidden_dims=dims)


@pytest.mark.parametrize("input_dim,latent_dim,name", [(0, 2, "input_dim"), (4, 0, "latent_dim")])
def test_decoder_requires_valid_ambient_and_latent_endpoints(input_dim, latent_dim, name):
    with pytest.raises(ValueError, match=f"{name}.*latent-to-ambient decoder"):
        Autoencoder(input_dim, [8, 6], latent_dim, decoder_hidden_dims=[7])

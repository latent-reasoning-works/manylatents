"""Encoder composition and real Lightning checkpoint lifecycle regressions."""

from unittest.mock import patch

import pytest
import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from manylatents.algorithms.lightning.mioflow import MIOFlow
from manylatents.algorithms.lightning.networks.autoencoder import Autoencoder
from manylatents.algorithms.lightning.networks.network import HasDecode, HasEncode

pytest.importorskip("torchdiffeq")
pytest.importorskip("ot")


class PopulationData:
    def __init__(self):
        generator = torch.Generator().manual_seed(12)
        self.batch = {
            "data": torch.randn(24, 4, generator=generator) * 2 + 3,
            "time": torch.tensor([2.0] * 12 + [5.0] * 12),
            "label": torch.arange(24) % 3,
        }

    def train_dataloader(self):
        return DataLoader([self.batch], batch_size=None)


class EncodeOnly(torch.nn.Module):
    latent_dim = 2

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 2)

    def encode(self, x):
        return self.projection(x)


def make_module(kind="gaga", datamodule=None):
    encoder = None
    if kind is not None:
        encoder = {
            "_target_": "manylatents.algorithms.lightning.networks."
            + ("gaga_net.GAGANetwork" if kind == "gaga" else "autoencoder.Autoencoder"),
            "input_dim": None,
            "latent_dim": 2,
            "hidden_dims": [8, 6],
            "activation": "tanh",
            "batchnorm": True,
            "dropout": 0.1,
        }
    return MIOFlow(
        network={
            "_target_": "manylatents.algorithms.lightning.networks.mioflow_net.MIOFlowODEFunc",
            "input_dim": None,
            "hidden_dim": 9,
            "init_seed": 17,
        },
        optimizer={"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.01},
        encoder=encoder,
        encoder_pretraining="gaga" if kind == "gaga" else "none",
        datamodule=datamodule,
        gaga_encoder_epochs=2,
        gaga_decoder_epochs=2,
        n_global_epochs=1,
        lambda_energy=0,
        n_bins=3,
        n_trajectories=4,
    )


def trainer(epochs=1):
    return Trainer(
        max_epochs=epochs, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )


def distances(self, data):
    # Isolate persistence from PHATE; its real fitting/learning is tested separately.
    return torch.cdist(torch.as_tensor(data), torch.as_tensor(data)).numpy()


def test_configure_model_twice_preserves_trained_modules():
    data = PopulationData()
    module = make_module("autoencoder", data)
    module.configure_model()
    flow, encoder, preprocessor = module.network, module.encoder, module.preprocessor
    initial = {k: v.clone() for k, v in flow.state_dict().items()}
    optimizer = module.configure_optimizers()
    loss = module._global_step(module._group_by_time(data.batch))["loss"]
    loss.backward()
    optimizer.step()
    trained = {k: v.clone() for k, v in flow.state_dict().items()}
    assert any(not torch.equal(initial[k], v) for k, v in trained.items())
    module.configure_model()
    module.setup("test")
    assert module.network is flow
    assert module.encoder is encoder
    assert module.preprocessor is preprocessor
    for key, value in trained.items():
        torch.testing.assert_close(module.network.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["gaga", "autoencoder", None])
def test_checkpoint_round_trip_without_setup_or_fitting(tmp_path, kind):
    data = PopulationData()
    module = make_module(kind, data)
    fit_trainer = trainer()
    with patch.object(MIOFlow, "_compute_gaga_target_distances", distances):
        fit_trainer.fit(module, train_dataloaders=data.train_dataloader())
    expected = module.encode(data.batch["data"])
    checkpoint = tmp_path / "model.ckpt"
    fit_trainer.save_checkpoint(checkpoint)
    state = {k: v.clone() for k, v in module.state_dict().items()}
    with (
        patch.object(MIOFlow, "setup", side_effect=AssertionError("setup during restore")),
        patch.object(MIOFlow, "_fit_gaga", side_effect=AssertionError("refitted encoder")),
        patch.object(MIOFlow, "_compute_gaga_target_distances", side_effect=AssertionError("recomputed statistics")),
    ):
        restored = MIOFlow.load_from_checkpoint(checkpoint, strict=True)
        restored.configure_model()
        restored.on_fit_start()
    assert restored.datamodule is None
    assert restored.ambient_dim == 4
    assert restored.latent_dim == (4 if kind is None else 2)
    assert restored._encoder_fitted.item()
    assert restored.state_dict().keys() == state.keys()
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(restored.encode(data.batch["data"]), expected, rtol=0, atol=0)
    torch.testing.assert_close(restored.encode(data.batch["data"], 2, 5), expected, rtol=0, atol=0)
    if kind is not None:
        assert restored.preprocessor.mean.shape == (4,)
        assert restored.preprocessor.std.shape == (4,)
        assert restored.preprocessor.dist_std.shape == ()
        assert not restored.encoder.training
        assert all(not p.requires_grad for p in restored.encoder.parameters())
    if kind == "gaga":
        torch.testing.assert_close(restored.preprocessor.mean, data.batch["data"].mean(0))
        assert not torch.equal(restored.preprocessor.std, torch.ones(4))
        # Exercise Trainer.fit(ckpt_path=...), where setup precedes state loading.
        resumed = make_module(kind, data)
        with patch.object(MIOFlow, "_fit_gaga", side_effect=AssertionError("refitted on resume")):
            trainer(epochs=2).fit(resumed, train_dataloaders=data.train_dataloader(), ckpt_path=checkpoint)
        for key, value in restored.encoder.state_dict().items():
            torch.testing.assert_close(resumed.encoder.state_dict()[key], value, rtol=0, atol=0)


def test_plain_autoencoder_swaps_in_and_flow_learns_fixed_coordinates():
    data = PopulationData()
    module = make_module("autoencoder", data)
    module.configure_model()
    assert isinstance(module.encoder, Autoencoder)
    assert isinstance(module.encoder, HasEncode)
    module.train()
    assert not module.encoder.training
    assert module.network.training
    coordinates = module._encode_coordinates(data.batch["data"].requires_grad_())
    assert coordinates.shape == (24, 2)
    assert not coordinates.requires_grad
    state = {k: v.clone() for k, v in module.encoder.state_dict().items()}
    optimizer = module.configure_optimizers()
    params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert params == {id(p) for p in module.network.parameters()}
    groups = module._group_by_time(data.batch)
    before = module._global_step(groups)["loss"].item()
    for _ in range(35):
        optimizer.zero_grad()
        module._global_step(groups)["loss"].backward()
        optimizer.step()
    after = module._global_step(groups)["loss"].item()
    assert after < before * 0.8, (before, after)
    for key, value in state.items():
        torch.testing.assert_close(module.encoder.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(module._encode_coordinates(data.batch["data"]), coordinates, rtol=0, atol=0)


def test_encode_only_contract_and_latent_trajectories():
    data = PopulationData()
    module = make_module(None, data)
    module.encoder_config = EncodeOnly()
    module.configure_model()
    assert isinstance(module.encoder, HasEncode)
    assert not isinstance(module.encoder, HasDecode)
    assert not module.supports_ambient_trajectories
    module._global_step(module._group_by_time(data.batch))["loss"].backward()
    assert any(p.grad is not None for p in module.network.parameters())
    assert all(p.grad is None for p in module.encoder.parameters())
    module._generate_trajectories()
    assert module.trajectories.shape == (3, 4, 2)


@pytest.mark.parametrize("kind", ["gaga", "autoencoder", None])
def test_ambient_trajectory_capability_before_running(kind):
    module = make_module(kind, PopulationData())
    if kind == "autoencoder":
        module.encoder_config["decoder_hidden_dims"] = [9, 7, 5]
    module.configure_model()
    assert module.trajectories is None
    # The query must not integrate, encode, decode, or fit anything.
    with patch.object(module.network, "forward", side_effect=AssertionError("ran flow")):
        if module.encoder is not None:
            assert isinstance(module.encoder, HasDecode)
            with patch.object(module.encoder, "decode", side_effect=AssertionError("decoded")):
                assert module.supports_ambient_trajectories
        else:
            assert module.supports_ambient_trajectories
    module._generate_trajectories()
    assert module.trajectories.shape == (3, 4, 4)


@pytest.mark.parametrize("decode", [None, 42])
def test_noncallable_decode_does_not_advertise_ambient_trajectories(decode):
    module = make_module(None, PopulationData())
    module.encoder_config = EncodeOnly()
    module.encoder_config.decode = decode
    module.configure_model()
    assert not module.supports_ambient_trajectories
    module._generate_trajectories()
    assert module.trajectories.shape == (3, 4, 2)


def test_trajectory_capability_requires_configuration():
    module = make_module("autoencoder", PopulationData())
    with pytest.raises(RuntimeError, match="Call configure_model"):
        _ = module.supports_ambient_trajectories


def test_explicit_time_wins_over_both_label_channels():
    data = PopulationData()
    data.batch["labels"] = torch.arange(24) % 4
    groups = make_module(None)._group_by_time(data.batch)
    assert [time for _, time in groups] == [2.0, 5.0]
    torch.testing.assert_close(groups[0][0], data.batch["data"][:12])
    torch.testing.assert_close(groups[1][0], data.batch["data"][12:])


def test_construction_does_not_fit_and_unfinished_stage_is_persisted():
    module = make_module("gaga", PopulationData())
    with patch.object(MIOFlow, "_fit_gaga", side_effect=AssertionError("fitted in construction")):
        module.setup("fit")
        module.configure_model()
    assert not module.state_dict()["_encoder_fitted"].item()
    assert torch.equal(module.preprocessor.mean, torch.zeros(4))


def test_hydra_configs_resolve_and_restore_without_their_parent(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "width": 7,
        "network": {
            "_target_": "manylatents.algorithms.lightning.networks.mioflow_net.MIOFlowODEFunc",
            "input_dim": None,
            "hidden_dim": "${width}",
        },
        "encoder": {
            "_target_": "manylatents.algorithms.lightning.networks.autoencoder.Autoencoder",
            "input_dim": None,
            "hidden_dims": ["${width}"],
            "decoder_hidden_dims": [9, "${width}", 5],
            "latent_dim": 2,
        },
    })
    data = PopulationData()
    module = MIOFlow(
        network=cfg.network,
        encoder=cfg.encoder,
        optimizer={"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.01},
        datamodule=data,
        lambda_energy=0,
        n_bins=3,
        n_trajectories=4,
    )
    fit_trainer = trainer()
    fit_trainer.fit(module, train_dataloaders=data.train_dataloader())
    assert cfg.network.input_dim is None
    assert cfg.encoder.input_dim is None
    assert module.hparams["network"]["hidden_dim"] == 7
    assert module.hparams["encoder"]["hidden_dims"] == [7]
    assert module.hparams["encoder"]["decoder_hidden_dims"] == [9, 7, 5]
    path = tmp_path / "hydra.ckpt"
    fit_trainer.save_checkpoint(path)
    restored = MIOFlow.load_from_checkpoint(path, strict=True)
    torch.testing.assert_close(restored.encode(data.batch["data"]), module.encode(data.batch["data"]))

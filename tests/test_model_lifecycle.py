"""Regression tests for config construction and Lightning's per-stage hooks."""

import functools
from unittest.mock import patch

import pytest
import torch
from lightning.pytorch import Callback, Trainer
from omegaconf import OmegaConf

from manylatents.algorithms.lightning.cflows import Cflows
from manylatents.algorithms.lightning.latent_ode import LatentODE
from manylatents.algorithms.lightning.mioflow import MIOFlow
from manylatents.algorithms.lightning.reconstruction import Reconstruction
from manylatents.data.precomputed_datamodule import PrecomputedDataModule


ALGORITHMS = [Reconstruction, LatentODE, Cflows, MIOFlow]
NETWORKS = "manylatents.algorithms.lightning.networks."


def network_config(algorithm):
    if algorithm is Reconstruction:
        return dict(_target_=NETWORKS + "autoencoder.Autoencoder", input_dim=2,
                    hidden_dims=[8], latent_dim=2, activation="tanh")
    if algorithm is MIOFlow:
        return dict(_target_=NETWORKS + "mioflow_net.MIOFlowODEFunc", input_dim=2,
                    hidden_dim=8)
    return dict(_target_=NETWORKS + "latent_ode.LatentODENetwork", input_dim=2,
                latent_dim=2, hidden_dim=8, encoder_hidden_dims=[],
                decoder_hidden_dims=[], ode_n_layers=1, solver="rk4",
                use_adjoint=False)


def make_model(algorithm, network, datamodule=None, seed=42):
    kwargs = dict(network=network, datamodule=datamodule, init_seed=seed,
                  optimizer=functools.partial(torch.optim.Adam, lr=0.03))
    if algorithm is MIOFlow:
        kwargs.update(lambda_energy=0, n_global_epochs=4, n_bins=3,
                      n_trajectories=4)
    elif algorithm is Cflows:
        kwargs.update(loss={"_target_": "manylatents.algorithms.lightning.losses.cflows.OTLoss"},
                      lambda_density=0)
    else:
        kwargs["loss"] = {"_target_": "manylatents.algorithms.lightning.losses.mse.MSELoss"}
    return algorithm(**kwargs)


def weights(network):
    return {name: value.detach().clone() for name, value in network.state_dict().items()}


def assert_weights_equal(expected, network):
    actual = network.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("config_type", [dict, OmegaConf.create], ids=["dict", "DictConfig"])
def test_configure_model_preserves_perturbed_parameters(algorithm, config_type):
    model = make_model(algorithm, config_type(network_config(algorithm)))
    model.configure_model()
    network = model.network
    parameters = tuple(network.parameters())
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(0.25)
    expected = weights(network)
    rng_state = torch.get_rng_state().clone()

    for _ in range(3):
        model.configure_model()
        assert_weights_equal(expected, model.network)
        assert model.network is network
        assert all(a is b for a, b in zip(parameters, model.network.parameters()))
        torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_config_initialization_ignores_prior_rng(algorithm):
    snapshots = []
    for prior_seed in (0, 99):
        torch.manual_seed(prior_seed)
        model = make_model(algorithm, network_config(algorithm), seed=42)
        model.configure_model()
        snapshots.append(weights(model.network))
    assert_weights_equal(snapshots[0], model.network)
    other = make_model(algorithm, network_config(algorithm), seed=43)
    other.configure_model()
    assert any(not torch.equal(value, other.network.state_dict()[name])
               for name, value in snapshots[0].items())


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_object_initialization_is_seeded_before_wrapping(algorithm):
    import hydra_zen

    snapshots = []
    for prior_seed in (0, 99):
        torch.manual_seed(prior_seed)
        network = hydra_zen.instantiate(network_config(algorithm), init_seed=42)
        initialized = weights(network)
        model = make_model(algorithm, network, seed=42)
        model.setup("fit")
        assert model.network is network
        assert_weights_equal(initialized, model.network)
        snapshots.append(initialized)
    assert_weights_equal(snapshots[0], model.network)

    config_model = make_model(algorithm, network_config(algorithm), seed=42)
    config_model.configure_model()
    assert_weights_equal(snapshots[0], config_model.network)


@pytest.mark.parametrize("variant", ["Vanilla", "VAE"])
def test_aanet_object_initialization_is_seeded_before_wrapping(variant):
    import hydra_zen

    cfg = dict(_target_=NETWORKS + "aanet." + variant, input_dim=2,
               layer_widths=[8], n_archetypes=3, device="cpu")
    snapshots = []
    for prior_seed in (0, 99):
        torch.manual_seed(prior_seed)
        network = hydra_zen.instantiate(cfg, init_seed=42)
        snapshots.append(weights(network))
        model = make_model(Reconstruction, network)
        model.setup("fit")
        assert model.network is network
        assert_weights_equal(snapshots[-1], network)
    assert_weights_equal(snapshots[0], network)
    config_model = make_model(Reconstruction, cfg)
    config_model.configure_model()
    assert_weights_equal(snapshots[0], config_model.network)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_supplied_weights_are_never_reset_to_wrapper_seed(algorithm):
    import hydra_zen

    # A caller can supply a pretrained or manually edited network, even if its
    # initialization seed differs from the wrapper's. The object is authoritative.
    network = hydra_zen.instantiate(network_config(algorithm))
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(0.25)
    expected = weights(network)
    model = make_model(algorithm, network, seed=123)
    for stage in ("fit", "validate", "test", "predict", None):
        model.setup(stage)
        model.configure_model()
        assert model.network is network
        assert_weights_equal(expected, model.network)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("config_type", [dict, OmegaConf.create], ids=["dict", "DictConfig"])
def test_setup_infers_input_dim_once(algorithm, config_type):
    cfg = network_config(algorithm)
    cfg["input_dim"] = None
    dm = PrecomputedDataModule(data=torch.ones(8, 2), batch_size=8)
    dm.setup()
    model = make_model(algorithm, config_type(cfg), dm)
    model.setup("fit")
    assert model.network_config["input_dim"] == 2
    network = model.network
    with patch.object(dm, "train_dataloader", side_effect=AssertionError("read data again")):
        model.setup("test")
        model.configure_model()
    assert model.network is network


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("construction", ["config", "object"])
def test_fit_test_extract_uses_trained_network(algorithm, construction):
    import hydra_zen

    # Two small populations with a learnable translation. One batch per epoch.
    x0 = torch.tensor([[-0.3, 0.1], [0.1, -0.2], [0.2, 0.3], [-0.1, -0.1]])
    x = torch.cat([x0, x0 + 1])
    time = torch.tensor([0.] * 4 + [1.] * 4)
    dm = PrecomputedDataModule(data=x, time=time, batch_size=8)
    dm.setup()
    cfg = OmegaConf.create(network_config(algorithm))
    supplied = hydra_zen.instantiate(cfg) if construction == "object" else None
    before_setup = weights(supplied) if supplied is not None else None
    model = make_model(algorithm, supplied if supplied is not None else cfg, dm)
    model.setup("fit")
    if supplied is not None:
        assert model.network is supplied
        assert_weights_equal(before_setup, supplied)
    model.eval()
    with torch.no_grad():
        initial_embeddings = model.encode(x).clone()
        if algorithm is Cflows:
            initial_gene_trajectory = model.gene_trajectory(x0, torch.linspace(0, 1, 3)).clone()
    initial_weights = weights(model.network)
    model.train()

    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=4, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, num_sanity_val_steps=0,
                      limit_val_batches=0)
    trainer.fit(model, datamodule=dm)
    model.eval()
    network = model.network
    trained_weights = weights(network)
    with torch.no_grad():
        trained_embeddings = model.encode(x).clone()
        if algorithm is Cflows:
            trained_gene_trajectory = model.gene_trajectory(x0, torch.linspace(0, 1, 3)).clone()
            assert (trained_gene_trajectory - initial_gene_trajectory).abs().max().item() > 1e-3
    assert (trained_embeddings - initial_embeddings).abs().max().item() > 1e-3
    assert any(not torch.equal(value, trained_weights[name])
               for name, value in initial_weights.items())
    trajectories = model.trajectories.clone() if algorithm is MIOFlow else None
    batch = next(iter(dm.test_dataloader()))
    with torch.no_grad(), patch.object(model, "log"), patch.object(model, "log_dict"):
        if algorithm is MIOFlow:
            trained_loss = model._global_step(model._group_by_time(batch))["loss"].item()
        else:
            trained_loss = model.shared_step(batch, 0, "test")["loss"].item()

    for _ in range(2):
        results = trainer.test(model, datamodule=dm, verbose=False)
        with torch.no_grad():
            extracted = model.encode(x)
        torch.testing.assert_close(extracted, trained_embeddings, rtol=0, atol=0)
        assert_weights_equal(trained_weights, model.network)
        assert model.network is network
        assert results[0]["test_loss"] == pytest.approx(trained_loss, rel=1e-6)
        if supplied is not None:
            assert model.network is supplied
        if algorithm is Cflows:
            with torch.no_grad():
                torch.testing.assert_close(
                    model.gene_trajectory(x0, torch.linspace(0, 1, 3)),
                    trained_gene_trajectory, rtol=0, atol=0,
                )
        if trajectories is not None:
            torch.testing.assert_close(model.trajectories, trajectories, rtol=0, atol=0)
            torch.testing.assert_close(model.encode(trajectories[0]), trajectories[-1],
                                       rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("algorithm", [Reconstruction, LatentODE, MIOFlow])
def test_run_experiment_returns_embeddings_captured_after_fit(algorithm):
    import numpy as np
    from manylatents.experiment import run_experiment

    x = torch.tensor([[-0.3, 0.1], [0.1, -0.2], [0.7, 1.1], [1.1, 0.8]])
    dm = PrecomputedDataModule(data=x, time=torch.tensor([0., 0., 1., 1.]), batch_size=4)
    model = make_model(algorithm, OmegaConf.create(network_config(algorithm)), dm)

    class CaptureEmbeddings(Callback):
        def on_train_start(self, trainer, pl_module):
            with torch.no_grad():
                self.initial = pl_module.encode(x).clone()
            pl_module.train()

        def on_fit_end(self, trainer, pl_module):
            pl_module.eval()
            with torch.no_grad():
                self.trained = pl_module.encode(x).clone()
            self.network = pl_module.network

    capture = CaptureEmbeddings()
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=4, logger=False,
                      callbacks=[capture], enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      num_sanity_val_steps=0, limit_val_batches=0)
    result = run_experiment(datamodule=dm, algorithm=model, trainer=trainer, seed=42)
    assert (capture.trained - capture.initial).abs().max().item() > 1e-3
    np.testing.assert_allclose(result["embeddings"], capture.trained.numpy(), rtol=0, atol=0)
    assert model.network is capture.network
    if algorithm is MIOFlow:
        trajectories = torch.as_tensor(result["trajectories"])
        torch.testing.assert_close(model.encode(trajectories[0]), trajectories[-1],
                                   rtol=1e-4, atol=1e-5)

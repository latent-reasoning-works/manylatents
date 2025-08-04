import pytest
from omegaconf import OmegaConf
from src.main import main

@pytest.mark.parametrize("cfg", [
    OmegaConf.create({
        "algorithms": [
            {"_target_": "src.algorithms.pca.PCAModule", "n_components": 2},
        ],
        "trainer": {
            "max_epochs": 1,
            "gpus": 0
        },
        "callbacks": {"embedding": {}},
        "data": {
            "_target_": "src.data.dummy.DummyDataModule",
            "batch_size": 32,
            "num_samples": 100,
            "input_dim": 10
        },
        "seed": 42,
        "debug": True,
        "project": "test",
        "eval_only": False,
        "cache_dir": "./cache"
    })
])
def test_main_pipeline(cfg):
    result = main(cfg)
    assert isinstance(result, dict)
    assert "embeddings" in result
    assert "metadata" in result
    assert "scores" in result 
"""Tests for algorithm name resolution."""
import pytest


def test_get_algorithm_by_name():
    from manylatents.algorithms.latent import get_algorithm
    cls = get_algorithm("pca")
    from manylatents.algorithms.latent.pca import PCAModule
    assert cls is PCAModule


def test_get_algorithm_case_insensitive():
    from manylatents.algorithms.latent import get_algorithm
    cls1 = get_algorithm("PCA")
    cls2 = get_algorithm("pca")
    assert cls1 is cls2


def test_get_algorithm_snake_case():
    from manylatents.algorithms.latent import get_algorithm
    cls = get_algorithm("diffusion_map")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule
    assert cls is DiffusionMapModule


def test_get_algorithm_collapsed_lowercase():
    from manylatents.algorithms.latent import get_algorithm
    cls = get_algorithm("diffusionmap")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule
    assert cls is DiffusionMapModule


def test_get_algorithm_class_name():
    from manylatents.algorithms.latent import get_algorithm
    cls = get_algorithm("PCAModule")
    from manylatents.algorithms.latent.pca import PCAModule
    assert cls is PCAModule


def test_get_algorithm_unknown_raises():
    from manylatents.algorithms.latent import get_algorithm
    with pytest.raises(KeyError, match="nonexistent"):
        get_algorithm("nonexistent")


def test_list_algorithms():
    from manylatents.algorithms.latent import list_algorithms
    names = list_algorithms()
    assert "pca" in names
    assert "umap" in names
    assert "diffusion_map" in names
    assert isinstance(names, list)
    # Should be sorted
    assert names == sorted(names)


def test_list_algorithms_no_duplicates():
    from manylatents.algorithms.latent import list_algorithms
    names = list_algorithms()
    assert len(names) == len(set(names))


def test_get_algorithm_hyphen_variant():
    from manylatents.algorithms.latent import get_algorithm
    cls = get_algorithm("diffusion-map")
    from manylatents.algorithms.latent.diffusion_map import DiffusionMapModule
    assert cls is DiffusionMapModule

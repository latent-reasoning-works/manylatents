"""Opt-in generator contracts and frozen pre-option regression results.

synthetic_original_arrays.npz holds the arrays, and synthetic_original_results.json
the global and dedicated RNG states, captured with the unchanged generators at
db669f0 on darwin-arm64. Random-stream states, integer arrays and the global
draw after generation are compared exactly on every platform. Float arrays are
compared with a tight tolerance only on the capture platform: BLAS results
differ between platforms, and Archetypal's SVD can flip signs on Linux even on
db669f0, so a stored float reference is not portable.
"""
import json
import platform
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

from manylatents.data import synthetic_dataset as synthetic

TREE = dict(graph_edges=[(0, 1, 1, 8), (1, 2, 2, 7), (1, 3, 3, 6)],
            n_dim=4, save_graph_viz=False, excluded_edges=[2])
CASES = {
    'SwissRoll': dict(n_distributions=3, n_points_per_distribution=[5, 7, 9], rotate_to_dim=5),
    'SwissRollGap': dict(n_distributions=3, n_points_per_distribution=[5, 7, 9], rotate_to_dim=5),
    'SaddleSurface': dict(n_distributions=3, n_points_per_distribution=[5, 7, 9], rotate_to_dim=5, use_gap=True, n_gaps=1),
    'Torus': dict(n_points=40, n_clusters=4, n_gaps=1, rotate_to_dim=5),
    'Archetypal': dict(n_obs=30, noise=0.2, output_dims=6, use_gap=True, n_gaps=1),
    'DLATreeFromGraph': TREE,
    'GaussianBlobs': dict(n_samples=30, centers=3, return_centers=True),
}


def snapshot(name, options):
    """Hash arrays and record both global and dedicated random-stream states."""
    streams = []
    factory = np.random.default_rng

    def tracked_rng(*args, **kwargs):
        rng = factory(*args, **kwargs)
        streams.append(rng)
        return rng

    np.random.seed(1234)
    with patch.object(np.random, 'default_rng', tracked_rng):
        obj = getattr(synthetic, name)(**options)
    arrays = [obj.data, obj.metadata, obj.get_gt_dists(), np.random.random(10)]
    if name == 'GaussianBlobs':
        arrays.append(obj.centers)
    return dict(arrays=[np.asarray(a) for a in arrays],
                streams=[rng.bit_generator.state for rng in streams])


CAPTURE_PLATFORM = 'darwin-arm64'


def assert_matches_original(name, result):
    here = Path(__file__).parent
    streams = json.loads((here / 'synthetic_original_results.json').read_text())[name]['streams']
    assert result['streams'] == streams
    with np.load(here / 'synthetic_original_arrays.npz') as original:
        expected = [original[f'{name}__{i}'] for i in range(len(result['arrays']))]
    np.testing.assert_array_equal(result['arrays'][3], expected[3])  # global RNG draw
    for got, want in zip(result['arrays'], expected):
        assert got.shape == want.shape and got.dtype == want.dtype
        if want.dtype.kind == 'f':
            if f'{sys.platform}-{platform.machine()}' == CAPTURE_PLATFORM:
                np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
        else:
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('name', CASES)
def test_defaults_match_frozen_original(name):
    options = CASES[name]
    assert_matches_original(name, snapshot(name, options))
    defaults = dict(min_separation=None) if name == 'GaussianBlobs' else dict(noise_mode='absolute')
    if name == 'DLATreeFromGraph':
        defaults['gt_metric'] = 'hops'
    assert_matches_original(name, snapshot(name, dict(options, **defaults)))


@pytest.mark.parametrize('name', list(CASES)[:-1])
@pytest.mark.parametrize('amount', [0.0, 0.2])
def test_relative_noise_scales_clean_cloud(name, amount):
    options = dict(CASES[name])
    knob = 'sigma' if name == 'DLATreeFromGraph' else 'noise'
    # Torus adds noise before excluding gaps; use all samples for this residual check.
    if name == 'Torus':
        options['n_gaps'] = 0
    def build(value, mode):
        np.random.seed(1234)
        return getattr(synthetic, name)(**dict(options, **{knob: value, 'noise_mode': mode}))
    clean = build(0, 'absolute')
    absolute = build(amount, 'absolute')
    relative = build(amount, 'relative')
    cloud = clean.M_complete if name == 'DLATreeFromGraph' else clean.data
    scale = np.sqrt(np.mean(np.sum((cloud - cloud.mean(axis=0)) ** 2, axis=1)))
    assert relative.noise_scale == pytest.approx(scale)
    assert relative.noise_std == pytest.approx(amount * scale)
    assert absolute.generation_info == dict(noise_mode='absolute', noise_scale=1.0, noise_std=amount)
    assert relative.generation_info == dict(noise_mode='relative', noise_scale=relative.noise_scale, noise_std=relative.noise_std)
    np.testing.assert_allclose(relative.data - clean.data, scale * (absolute.data - clean.data), atol=1e-13)
    np.testing.assert_array_equal(relative.metadata, clean.metadata)
    np.testing.assert_array_equal(relative.get_gt_dists(), clean.get_gt_dists())


@pytest.mark.parametrize('name', list(CASES)[:-1])
def test_invalid_noise_mode(name):
    with pytest.raises(ValueError, match='absolute.*relative'):
        getattr(synthetic, name)(**CASES[name], noise_mode='invalid')


@pytest.mark.parametrize('centers', [None, 5, np.int64(4)])
def test_separated_centers_and_point_sampling(centers):
    options = dict(n_samples=100, n_features=3, centers=centers, min_separation=4.0, random_state=17)
    first = synthetic.GaussianBlobs(**options, cluster_std=0.2)
    second = synthetic.GaussianBlobs(**options, cluster_std=3.0, return_centers=True)
    assert pdist(first.centers).min() >= 4.0
    np.testing.assert_array_equal(first.centers, second.centers)
    expected, labels = make_blobs(n_samples=100, n_features=3, centers=first.centers, cluster_std=0.2, random_state=17)
    np.testing.assert_array_equal(first.data, expected)
    np.testing.assert_array_equal(first.metadata, labels)


def test_impossible_separation():
    with pytest.raises(ValueError, match=r'^min_separation_unsatisfiable:.*centers.*n_features.*center_box.*min_separation'):
        synthetic.GaussianBlobs(centers=3, n_features=2, center_box=(0, 1), min_separation=5)


def test_separation_rejects_explicit_centers():
    with pytest.raises(ValueError, match='centers'):
        synthetic.GaussianBlobs(centers=[[0, 0], [1, 1]], min_separation=1)


@pytest.mark.measurement
def test_separation_components():
    print('\nratio centers components')
    for ratio in (4, 6, 8):
        for centers in range(3, 8):
            obj = synthetic.GaussianBlobs(n_samples=1000, n_features=3, centers=centers,
                cluster_std=1.0, min_separation=float(ratio), center_box=(-10*ratio, 10*ratio), random_state=42)
            graph = NearestNeighbors(n_neighbors=15).fit(obj.data).kneighbors_graph()
            count = connected_components(graph.maximum(graph.T), directed=False, return_labels=False)
            print(f'{ratio:5d} {centers:7d} {count:10d}')
            if ratio == 8:
                assert count == centers


@pytest.mark.parametrize('include_gaps', [False, True])
def test_clean_length_topology_zero_edges_and_noise_independence(include_gaps):
    clean = synthetic.DLATreeFromGraph(**TREE, sigma=0, gt_metric='clean_length')
    noisy = synthetic.DLATreeFromGraph(**TREE, sigma=2, gt_metric='clean_length', noise_mode='relative')
    np.testing.assert_array_equal(clean.M_complete_clean, noisy.M_complete_clean)
    weights = clean.adj_matrix_complete.copy()
    rows, cols = weights.nonzero()
    lengths = np.linalg.norm(clean.M_complete[rows] - clean.M_complete[cols], axis=1)
    assert np.any(lengths == 0), 'Shared boundary samples must exercise zero-length edges'
    # Sparse explicit zero edges supply an independent exact reference.
    weights.data = lengths
    expected = shortest_path(weights, directed=False)
    if not include_gaps:
        expected = expected[np.ix_(clean.visible_indices, clean.visible_indices)]
    actual = clean.get_gt_dists(include_gaps=include_gaps)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual, noisy.get_gt_dists(include_gaps=include_gaps))
    np.testing.assert_array_equal(clean.adj_matrix_clean_length.toarray() != 0, clean.adj_matrix_complete.toarray() != 0)
    assert np.isfinite(actual).all()


def test_invalid_gt_metric():
    with pytest.raises(ValueError, match='hops.*clean_length'):
        synthetic.DLATreeFromGraph(**TREE, gt_metric='invalid')


@pytest.mark.parametrize('mode', ['full', 'split'])
@pytest.mark.parametrize('module,cls,options,expected', [
    ('swissroll', 'SwissRollDataModule', dict(CASES['SwissRoll']), synthetic.SwissRoll),
    ('swissroll', 'SwissRollDataModule', dict(CASES['SwissRollGap'], use_gap=True), synthetic.SwissRollGap),
    ('saddlesurface', 'SaddleSurfaceDataModule', CASES['SaddleSurface'], synthetic.SaddleSurface),
    ('torus', 'TorusDataModule', CASES['Torus'], synthetic.Torus),
    ('archetypal', 'ArchetypalDataModule', CASES['Archetypal'], synthetic.Archetypal),
    ('dlatree', 'DLATreeFromGraphDataModule', dict(TREE, gt_metric='clean_length'), synthetic.DLATreeFromGraph),
    ('gaussian_blobs', 'GaussianBlobDataModule', dict(CASES['GaussianBlobs'], min_separation=4), synthetic.GaussianBlobs),
])
def test_wrappers_forward_options(module, cls, options, expected, mode):
    import importlib
    options = dict(options)
    if expected is not synthetic.GaussianBlobs:
        options['noise_mode'] = 'relative'
    wrapper = getattr(importlib.import_module('manylatents.data.' + module), cls)(**options, mode=mode)
    wrapper.setup()
    obj = wrapper.train_dataset if mode == 'full' else wrapper.dataset
    assert isinstance(obj, expected)
    if expected is synthetic.GaussianBlobs:
        assert pdist(obj.centers).min() >= 4
    else:
        assert obj.generation_info['noise_mode'] == 'relative'
    if expected is synthetic.DLATreeFromGraph:
        assert obj.gt_metric == 'clean_length'


def test_torus_scale_uses_complete_cloud_before_gaps():
    obj = synthetic.Torus(n_points=120, n_clusters=4, n_gaps=2, noise=0.3, noise_mode='relative')
    rng = np.random.default_rng(42)
    theta = 2 * np.pi * rng.random(240)
    phi = 2 * np.pi * rng.random(240)
    cloud = np.column_stack(((3 + np.cos(phi)) * np.cos(theta),
                             (3 + np.cos(phi)) * np.sin(theta), np.sin(phi)))
    scale = np.sqrt(np.mean(np.sum((cloud - cloud.mean(axis=0)) ** 2, axis=1)))
    assert obj.noise_scale == pytest.approx(scale)
    assert obj.noise_std == pytest.approx(0.3 * scale)


def test_gap_static_generator_preserves_return_contract():
    options = dict(n_distributions=3, n_points_per_distribution=[5, 7, 9])
    arrays = synthetic.SwissRollGap.generate_swissroll_gap(**options, noise_mode='relative')
    obj = synthetic.SwissRollGap(**options, noise_mode='relative')
    assert len(arrays) == 3
    for actual, expected in zip(arrays, (obj.data, obj.data_gt, obj.metadata)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_separation_rejects_invalid_distance(value):
    with pytest.raises(ValueError, match='min_separation'):
        synthetic.GaussianBlobs(min_separation=value)


@pytest.mark.parametrize('filename', ['swissroll', 'saddle_surface', 'torus', 'archetypal',
                                     'dla_tree_from_graph', 'dla_tree_from_graph_nogaps', 'gaussian_blobs'])
def test_configs_expose_default_options(filename):
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    import inspect
    config = OmegaConf.load(Path(__file__).parents[1] / 'manylatents/configs/data' / (filename + '.yaml'))
    options = {'min_separation': None} if filename == 'gaussian_blobs' else {'noise_mode': 'absolute'}
    if filename.startswith('dla_tree_from_graph'):
        options['gt_metric'] = 'hops'
    signature = inspect.signature(get_class(config['_target_']))
    for name, default in options.items():
        assert config[name] == default
        assert signature.parameters[name].default == default


def test_separation_list_n_samples_uses_one_center_per_entry():
    blobs = synthetic.GaussianBlobs(n_samples=[50, 60, 70, 80], n_features=3,
                                    min_separation=4.0, center_box=(-40.0, 40.0))
    assert blobs.centers.shape == (4, 3)
    assert np.bincount(blobs.metadata).tolist() == [50, 60, 70, 80]

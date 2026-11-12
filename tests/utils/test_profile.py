import numpy as np
import pytest
from scipy.spatial.distance import pdist
from manylatents.utils.profile import point_cloud_pca, materialize_profile_input


@pytest.mark.parametrize('n,d', [(80,1),(80,2),(80,5),(80,50),(80,51),(3,10)])
def test_pca_preserves_distances_when_width_at_most_50(n, d):
    x = np.random.default_rng(3).normal(size=(n,d))
    y = point_cloud_pca(x)
    assert y.shape == (n,min(50,d)) and y.dtype == np.float64
    np.testing.assert_array_equal(y, point_cloud_pca(x))
    if d <= 50:
        np.testing.assert_allclose(pdist(x), pdist(y), atol=1e-12)


def test_materialize_forwards_seed_and_kwargs_once(monkeypatch):
    from manylatents import api
    calls = []
    x = np.arange(12.).reshape(4,3)
    class Module:
        def setup(self):
            calls.append('setup')
            self.train_dataset = type('Dataset', (), {'data': x})()
    def resolve(**kwargs):
        calls.append(kwargs)
        return Module()
    monkeypatch.setattr(api, '_resolve_datamodule', resolve)
    np.testing.assert_array_equal(materialize_profile_input('cloud', seed=12, data_kwargs={'n':4}), x)
    assert calls == [{'data':'cloud','seed':12,'n':4}, 'setup']

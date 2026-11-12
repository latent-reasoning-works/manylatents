"""The engine can expose its loaded matrix without another data load."""
import numpy as np
from manylatents.api import run


def test_return_input_data_is_opt_in():
    x = np.arange(120, dtype=np.float32).reshape(30,4)
    ordinary = run(input_data=x, algorithms={'latent':'pca'}, n_components=2)
    assert 'input_data' not in ordinary
    result = run(input_data=x, algorithms={'latent':'pca'}, n_components=2,
                 return_input_data=True)
    np.testing.assert_array_equal(result['input_data'], x)

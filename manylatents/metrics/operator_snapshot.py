"""Copy the actual post-fit neighborhood objects without fitting or repairing them.

Use ``at='embedding'`` in Hydra metric configurations. The evaluator still passes
``module``, while ``at='module'`` would trigger unrelated spectral prewarming.
The t-SNE reference is captured after the API-triggered transform has recentered
it. PHATE potential coordinates are never interpreted as transition weights.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np

from manylatents.metrics.registry import register_metric


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if hasattr(value, 'toarray'):
        value = value.toarray()
    result = np.array(value, copy=True)
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError('A required captured array is nonnumeric or nonfinite')
    return result


def _hash(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(array.dtype.str.encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _describe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return {'unavailable': 'NONFINITE_PARAMETER', 'repr': repr(value)}
        return value
    if isinstance(value, np.generic):
        return _describe(value.item())
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value]
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': _hash(value)}
    return {'class': f'{type(value).__module__}.{type(value).__name__}',
            'repr': repr(value)[:200]}


def _source(obj):
    try:
        filename = inspect.getsourcefile(obj)
        if filename is None:
            return {'status': 'SOURCE_UNAVAILABLE'}
        path = Path(filename).resolve()
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': 'OK'}
    except (TypeError, OSError):
        return {'status': 'SOURCE_UNAVAILABLE'}


def _storage(value):
    if hasattr(value, 'toarray'):
        data = np.asarray(value.data)
        return {'representation': value.format, 'shape': list(value.shape),
                'stored_entries': int(value.nnz), 'explicit_zero_entries': int(np.sum(data == 0)),
                'nonzero_entries': int(np.count_nonzero(data)), 'dtype': str(value.dtype)}
    array = np.asarray(value)
    return {'representation': 'dense', 'shape': list(array.shape),
            'stored_entries': int(array.size), 'explicit_zero_entries': int(np.sum(array == 0)),
            'nonzero_entries': int(np.count_nonzero(array)), 'dtype': str(array.dtype)}


@register_metric(aliases=['operator_snapshot'], default_params={},
                 description='Raw post-fit neighborhood and layout stage snapshots')
def OperatorSnapshot(embeddings, dataset=None, module=None, cache=None):
    """Return copied raw arrays and metadata for the three audited CPU modules.

    No new fit, transform, kernel reconstruction, graph normalization, or matrix
    power is performed. PHATE's potential must already have been calculated by
    the native transform. Arrays retain original dtypes, except the explicitly
    named float64 ``input_effective`` upcast. Unsupported paths raise.
    """
    from manylatents.algorithms.latent.umap import UMAPModule
    from manylatents.algorithms.latent.tsne import TSNEModule
    from manylatents.algorithms.latent.phate import PHATEModule

    if module is None or not getattr(module, '_is_fitted', False):
        raise ValueError('OperatorSnapshot requires a fitted native module')
    if getattr(module, '_resolved_backend', 'UNAVAILABLE') is not None:
        raise ValueError('OperatorSnapshot supports only the audited CPU backend')
    if getattr(module, 'fit_fraction', None) != 1:
        raise ValueError('Snapshot particle correspondence requires fit_fraction=1')
    if dataset is None or not hasattr(dataset, 'data'):
        raise ValueError('Snapshot requires the actual input dataset.data')
    X, Y = _array(dataset.data), _array(embeddings)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError('Input and returned layout must have matching particle rows')
    if Y.shape[1] != module.n_components:
        raise ValueError('Returned layout dimension differs from the native module')
    n = len(X)
    fields = ['n_components', 'random_state', 'init_seed', 'fit_fraction', '_resolved_backend',
              'backend', 'device', 'neighborhood_size']
    result = {'input_data': X, 'input_effective': X.astype(np.float64), 'returned_embedding': Y}
    sources = {'snapshot': _source(OperatorSnapshot), 'native_module': _source(type(module))}
    if isinstance(module, UMAPModule):
        method = 'umap'
        backend = module.model
        if type(backend).__module__ != 'umap.umap_' or type(backend).__name__ != 'UMAP':
            raise ValueError('Unexpected UMAP backend class')
        graph = backend.graph_
        fields += ['n_neighbors', 'min_dist', 'n_epochs', 'learning_rate', 'negative_sample_rate', 'metric']
        stage = 'stored_graph_after_optimizer_preparation'
        result['backend_embedding_after_transform'] = _array(backend.embedding_)
        extra = {key: _describe(getattr(backend, key, None))
                 for key in ('_n_neighbors', '_small_data', '_sigmas', '_rhos', '_a', '_b')}
        params = backend.get_params(deep=False)
        package_names = ('umap-learn', 'scipy')
    elif isinstance(module, TSNEModule):
        method = 'tsne'
        backend = module.affinities
        if type(backend).__module__ != 'openTSNE.affinity' or type(backend).__name__ != 'PerplexityBasedNN':
            raise ValueError('Unexpected t-SNE affinity class')
        graph = backend.P
        fields += ['perplexity', 'n_iter_early', 'n_iter_late', 'learning_rate', 'initialization', 'metric']
        stage = 'symmetrized_globally_normalized_joint_probability'
        reference = _array(module.embedding_train)
        if reference.shape != Y.shape:
            raise ValueError('t-SNE reference and returned layout have different row identities or shape')
        result['optimized_reference_after_transform'] = reference
        extra = {key: _describe(getattr(backend, key, None))
                 for key in ('perplexity', 'effective_perplexity_', 'symmetrize', 'n_jobs')}
        extra['knn_index_class'] = f'{type(backend.knn_index).__module__}.{type(backend.knn_index).__name__}'
        extra['knn_index_k'] = _describe(getattr(backend.knn_index, 'k', None))
        sources['backend_embedding'] = _source(type(module.embedding_train))
        try:
            signature = inspect.signature(type(module.embedding_train).transform)
            transform_defaults = {key: _describe(param.default) for key, param in signature.parameters.items()
                                  if param.default is not inspect.Parameter.empty}
        except (ValueError, TypeError, AttributeError):
            transform_defaults = {'status': 'UNAVAILABLE'}
        extra['transform_defaults'] = transform_defaults
        params = dict(extra)
        package_names = ('openTSNE',)
    elif isinstance(module, PHATEModule):
        method = 'phate'
        backend = module.model
        if type(backend).__module__ != 'phate.phate' or type(backend).__name__ != 'PHATE':
            raise ValueError('Unexpected PHATE backend class')
        if module.n_landmark is not None or backend.n_landmark is not None:
            raise ValueError('Landmark PHATE is outside particle-level snapshot scope')
        if 'Landmark' in type(backend.graph).__name__:
            raise ValueError('Landmark graph changes particle correspondence')
        if getattr(backend, '_diff_potential', None) is None:
            raise ValueError('PHATE potential was not computed by the native transform')
        graph = backend.graph.K
        fields += ['knn', 't', 'gamma', 'decay', 'n_landmark', 'n_pca', 'n_jobs', 'verbose']
        stage = 'stored_symmetrized_graph_kernel'
        # Read the public getter only after confirming its cached value exists.
        potential = _array(backend.diff_potential)
        if potential.shape != (n, n):
            raise ValueError('PHATE potential is not indexed by the same n particles')
        result['phate_potential'] = potential
        result['backend_embedding_after_transform'] = _array(backend.embedding)
        extra = {'t': _describe(backend.t), 'optimal_t': _describe(backend.optimal_t),
                 'gamma': _describe(backend.gamma), 'potential_log_offset': 1e-7 if backend.gamma == 1 else None,
                 'graph_class': f'{type(backend.graph).__module__}.{type(backend.graph).__name__}',
                 'graph_parameters': _describe(backend.graph.get_params())}
        params = backend.get_params()
        sources['backend_graph'] = _source(type(backend.graph))
        package_names = ('phate', 'graphtools')
    else:
        raise ValueError(f'Unsupported native module {type(module).__name__}')
    K = _array(module.kernel(ignore_diagonal=False))
    P = _array(module.affinity(ignore_diagonal=False, use_symmetric=False))
    if K.shape != (n, n) or P.shape != (n, n):
        raise ValueError('Captured matrices must have exactly n particle rows and columns')
    if np.any(K < 0) or np.any(P < 0):
        raise ValueError('Captured neighborhood weights must be nonnegative')
    result.update(kernel=K, affinity=P)
    sources['backend'] = _source(type(backend))
    packages = {}
    for package in ('numpy', 'torch', 'manylatents') + package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result['metadata'] = dict(
        method=method, capture_stage='embedding',
        capture_stage_evidence='required metric routing contract; at is not passed to this function',
        optimized_reference_class=(f'{type(module.embedding_train).__module__}.{type(module.embedding_train).__name__}'
                                   if method == 'tsne' else None),
        native_class=f'{type(module).__module__}.{type(module).__name__}',
        backend_class=f'{type(backend).__module__}.{type(backend).__name__}',
        kernel_stage=stage, native_affinity_stage='native_affinity_with_self_retained',
        returned_layout_stage='api_fit_then_transform_output',
        optimized_reference_stage='optimized_reference_after_transform' if method == 'tsne' else None,
        tSNE_reference_gauge='inplace_bounding_box_recentering_before_query_placement' if method == 'tsne' else None,
        self_policy='retained; physical escape masks must include focal identity',
        array_identities={key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': _hash(value)}
                          for key, value in result.items()},
        native_graph_storage=_storage(graph),
        module_parameters={key: _describe(getattr(module, key)) for key in fields},
        backend_parameters=_describe(params), realized=extra, source_files=sources, packages=packages,
    )
    return result

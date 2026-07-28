from typing import Any, Dict, Optional, Protocol, Tuple, Union

import numpy as np

from manylatents.algorithms.latent.latent_module_base import LatentModule


class Metric(Protocol):
    """Protocol for metrics that evaluate embeddings.

    .. note::
       This protocol is **documentation, not enforcement**. It is not
       ``@runtime_checkable``, nothing annotates against it, and metrics are plain functions
       rather than classes — so an ``isinstance`` check would raise, and a member check would
       reduce to ``callable()``. It is kept because the return contract below is genuinely
       load-bearing and was previously written down nowhere; the enforcement that exists lives
       in :func:`manylatents.metrics.registry._to_scalar`, at the one place a return value is
       actually reduced.

    A metric is a callable that takes embeddings as input and returns one of:

    1. float: Simple scalar metric (e.g., 0.95)
    2. tuple[float, np.ndarray]: Scalar first, per-sample values second. The FIRST element is
       what gets recorded, so it must be the summary — not, say, the per-sample array with a
       count bolted on.
    3. dict[str, Any]: Structured output. **If more than one value is numeric, the metric MUST
       declare which key is the measurement**, via ``scalar_key=`` on its ``@register_metric``.
       Otherwise ``_to_scalar`` raises rather than picking one.

       This rule exists because it was previously absent. The registry reduced a dict by
       taking whichever key came first — insertion order, i.e. whatever the author happened to
       write first — so ``connected_components`` recorded 66.667 (the mean component *size*)
       for a 3-component graph under a description promising the count, and the recorded value
       grew as the true count fell. ``shepard_residual`` recorded the OLS slope, and
       ``topology_descriptor`` recorded ``n_samples``.

    A metric must also be a *measurement*: returning a constant for structurally different
    inputs is a defect even when the constant is defensible. Four cross-modal metrics returned
    exactly 1.0 for any single array — true as self-similarity, useless in a declared suite,
    where nothing downstream can tell a constant from a measurement. Raise instead.

    Standard parameters:
        embeddings: Low-dimensional embedding array (n_samples, n_dims)
        dataset: Dataset object with .data attribute for high-dimensional data
        module: Fitted LatentModule instance (for accessing affinity matrices, etc.)

    Cache parameter:
        cache: Optional dict shared across metrics within one evaluation run.
            Metrics should pass this through to compute_knn() and
            compute_eigenvalues() from manylatents.utils.metrics.
            Do NOT slice the cache directly — call the utility functions,
            which handle cache lookup and population internally.

            If None, utility functions compute from scratch (backward compatible).
    """
    def __call__(
        self,
        embeddings: np.ndarray,
        dataset: Optional[object] = None,
        module: Optional[LatentModule] = None,
        cache: Optional[dict] = None,
    ) -> Union[float, Tuple[float, np.ndarray], dict[str, Any]]:
        ...

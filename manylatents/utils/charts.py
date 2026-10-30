"""Charts over a point cloud: local neighbourhoods that together cover it."""
from typing import NamedTuple

import numpy as np

from manylatents.utils.knn import compute_knn


class BallCover(NamedTuple):
    """A cover of a point cloud by kNN balls.

    Attributes:
        landmarks: ``(L,)`` int array of landmark row indices, in selection order.
            ``landmarks[0] == 0``.
        members: ``(L, min(m, n))`` int array. Row ``i`` holds the members of
            the ball around ``landmarks[i]``: the landmark itself in column 0,
            then its nearest neighbours by increasing distance.
    """

    landmarks: np.ndarray
    members: np.ndarray


def ball_cover(X, m: int, cache: dict | None = None) -> BallCover:
    """Cover ``X`` with balls of its ``m`` nearest neighbours around landmarks.

    Landmarks are chosen by farthest-point sampling restricted to uncovered
    points. The first landmark is row 0. Each next landmark is the uncovered
    point farthest (squared Euclidean) from all landmarks so far. Its ``m``
    nearest neighbours become covered. Selection stops when every row is
    covered. The result is deterministic for a given ``X``, ``m`` and kNN
    backend; ties in ``argmax`` go to the lowest row index.

    Args:
        X: ``(n, d)`` real array. float32 and float64 are both accepted.
        m: Ball size. It counts the landmark itself, so ``m=1`` makes every
            point its own landmark. ``m >= n`` gives a single ball holding
            every row.
        cache: Optional dict passed to :func:`compute_knn`. Use one dict per
            ``X``. Calling first with the largest ``m`` lets every smaller ``m``
            slice the stored neighbours instead of searching again.

    Returns:
        A :class:`BallCover` of landmark indices and a 2-D member array.
    """
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 1:
        raise ValueError(f"m must be a positive integer; got {m!r}")
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"X must be a nonempty 2-D array; got shape {X.shape}")

    n = X.shape[0]
    m = min(int(m), n)
    if m == 1:
        idx = np.arange(n, dtype=np.intp)
        nn = idx[:, None]
    else:
        # compute_knn's k excludes self; include_self puts self in column 0.
        _, nn = compute_knn(X, k=m - 1, include_self=True, cache=cache)
        nn = nn.astype(np.intp, copy=False)

    X64 = X.astype(np.float64)
    sq = (X64 * X64).sum(1)
    tmp = np.empty(n)
    covered = np.zeros(n, dtype=bool)
    score = ((X64 - X64[0]) ** 2).sum(1)
    covered[nn[0]] = True
    score[nn[0]] = -1
    ncov = int(covered.sum())
    landmarks = [0]
    while ncov < n:
        j = int(score.argmax())
        landmarks.append(j)
        new = nn[j][~covered[nn[j]]]
        covered[new] = True
        ncov += len(new)
        # Squared distance to X[j], expanded: |x|^2 - 2 x.xj + |xj|^2.
        np.dot(X64, X64[j], out=tmp)
        tmp *= -2
        tmp += sq
        tmp += sq[j]
        np.minimum(score, tmp, out=score)
        score[covered] = -1

    landmarks = np.asarray(landmarks, dtype=np.intp)
    return BallCover(landmarks=landmarks, members=nn[landmarks])

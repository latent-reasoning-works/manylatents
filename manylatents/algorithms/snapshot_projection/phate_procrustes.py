"""PHATE+Procrustes target projection for cross-dim distillation."""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from manylatents.lightning.activation_snapshot import ActivationSnapshot

__all__ = [
    "pca_reduce",
    "procrustes_align",
    "project_to_student_dim",
]


def pca_reduce(x: Tensor, n_components: int) -> Tensor:
    """Project ``x`` (n, d) onto its top-``n_components`` principal directions.

    Returned shape is ``(n, k)`` where ``k = min(n_components, n, d)``. Always
    operates in float32 on CPU; caller is responsible for restoring dtype/device.
    """
    if x.ndim != 2:
        raise ValueError(f"pca_reduce expects (n, d) input, got shape {tuple(x.shape)}")
    x32 = x.to(torch.float32).cpu()
    centered = x32 - x32.mean(dim=0, keepdim=True)
    u, s, _vt = torch.linalg.svd(centered, full_matrices=False)
    k = min(n_components, u.shape[1], s.shape[0])
    return u[:, :k] * s[:k]


def _center_scale(x: Tensor) -> tuple[Tensor, Tensor, float]:
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    norm = float(torch.linalg.norm(centered).item())
    if norm < 1e-12:
        norm = 1.0
    return centered / norm, mean, norm


def procrustes_align(source: Tensor, target: Tensor) -> Tensor:
    """Rotate ``source`` (n, d_low) into the coordinate frame of ``target`` (n, d).

    If ``target`` has a wider second dim, it is PCA-reduced to ``source``'s
    dim before alignment. Returns a tensor of shape ``(n, d_low)`` in float32.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("procrustes_align expects 2-D source and target")
    n = min(source.shape[0], target.shape[0])
    if n < 2:
        raise ValueError("procrustes_align needs at least 2 paired samples")

    src = source[:n].to(torch.float32).cpu()
    tgt = target[:n].to(torch.float32).cpu()

    target_dim = src.shape[1]
    if tgt.shape[1] != target_dim:
        tgt = pca_reduce(tgt, n_components=target_dim)

    src_normed, _, _ = _center_scale(src)
    tgt_normed, tgt_mean, tgt_scale = _center_scale(tgt)

    u, _s, vt = torch.linalg.svd(src_normed.T @ tgt_normed, full_matrices=False)
    rotation = u @ vt
    return (src_normed @ rotation) * tgt_scale + tgt_mean


def project_to_student_dim(
    snapshot: ActivationSnapshot,
    student_hidden_dim: int,
    *,
    knn: int = 5,
    t: Union[int, str] = "auto",
    decay: int = 40,
    gamma: float = 1.0,
    fit_fraction: float = 1.0,
    n_pca: Optional[int] = None,
    n_landmark: Optional[int] = None,
    n_jobs: int = -1,
    procrustes_align_to_teacher: bool = True,
    random_state: Optional[int] = None,
) -> ActivationSnapshot:
    """Replace each per-layer activation with a student-dim aligned target via
    PHATE, then optionally rotate into the PCA-reduced teacher's frame.

    Preserves ``input_ids``, ``attention_mask``, ``sample_ids``, and
    ``reduction``. Per-layer dtype and device are restored on the way out.
    """
    from manylatents.algorithms.latent.phate import PHATEModule

    new_activations: dict[str, Tensor] = {}
    for layer_path, acts in snapshot.activations.items():
        teacher_acts = acts.detach().to(torch.float32).cpu()

        phate = PHATEModule(
            n_components=student_hidden_dim,
            random_state=random_state,
            knn=int(knn),
            t=t,
            decay=int(decay),
            gamma=float(gamma),
            fit_fraction=float(fit_fraction),
            n_pca=n_pca,
            n_landmark=n_landmark,
            n_jobs=int(n_jobs),
        )
        z_phate = phate.fit_transform(teacher_acts)
        if not isinstance(z_phate, Tensor):
            z_phate = torch.as_tensor(z_phate)

        if procrustes_align_to_teacher:
            target = procrustes_align(z_phate, teacher_acts)
        else:
            target = z_phate.to(torch.float32)

        new_activations[layer_path] = target.to(acts.device).to(acts.dtype)

    return ActivationSnapshot(
        input_ids=snapshot.input_ids,
        attention_mask=snapshot.attention_mask,
        sample_ids=list(snapshot.sample_ids),
        activations=new_activations,
        reduction=snapshot.reduction,
    )

"""Project an ActivationSnapshot onto a different hidden dimensionality.

PHATE-based pipeline that produces a student-dim aligned target snapshot from a
teacher-dim snapshot. Used to make MSE-style alignment losses well-defined when
teacher and student have different hidden sizes.
"""
from manylatents.algorithms.snapshot_projection.phate_procrustes import (
    pca_reduce,
    procrustes_align,
    project_to_student_dim,
)

__all__ = [
    "pca_reduce",
    "procrustes_align",
    "project_to_student_dim",
]

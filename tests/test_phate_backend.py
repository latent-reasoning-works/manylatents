"""Tests for PHATEModule backend routing."""
import numpy as np
import pytest
import torch

from manylatents.utils.backend import check_torchdr_available

TORCHDR_AVAILABLE = check_torchdr_available()


def test_phate_default_backend_unchanged():
    """PHATEModule with default backend uses phate library."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(n_components=2, random_state=42, knn=5, t=5)
    x = torch.randn(50, 10)
    m.fit(x)
    assert m._is_fitted
    from phate import PHATE
    assert isinstance(m.model, PHATE)


def test_phate_accepts_backend_param():
    """PHATEModule accepts backend/device without error."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(n_components=2, random_state=42, backend=None, device=None)
    assert m.backend is None


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_torchdr_backend_fit_transform():
    """PHATEModule with torchdr backend produces embeddings."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(
        n_components=2, random_state=42, knn=5, t=5,
        backend="torchdr", device="cpu",
    )
    x = torch.randn(50, 10)
    emb = m.fit_transform(x)
    assert emb.shape == (50, 2)


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_torchdr_param_mapping():
    """PHATEModule maps knn->k and decay->alpha for TorchDR."""
    from manylatents.algorithms.latent.phate import PHATEModule

    m = PHATEModule(
        n_components=2, random_state=42, knn=10, t=50, decay=40,
        backend="torchdr", device="cpu",
    )
    # TorchDR PHATE uses 'k' not 'knn', 'alpha' not 'decay'
    assert m.model.k == 10
    assert m.model.alpha == 40
    assert m.model.t == 50


@pytest.mark.skipif(not TORCHDR_AVAILABLE, reason="torchdr not installed")
def test_phate_backend_output_agreement():
    """CPU and TorchDR PHATE produce structurally consistent affinity matrices.

    Both backends should produce:
    - NxN symmetric affinity matrices
    - Non-negative values
    - Leading eigenvalue == 1.0
    - Kernel matrix with max == 1.0 (self-affinity)
    - Eigenvalue spectrum in [0, 1]
    """
    from manylatents.algorithms.latent.phate import PHATEModule

    x = torch.randn(80, 15, generator=torch.Generator().manual_seed(42))

    # CPU PHATE
    m_cpu = PHATEModule(n_components=2, random_state=42, knn=10, t=5, backend=None)
    emb_cpu = m_cpu.fit_transform(x)
    A_cpu = m_cpu.affinity_matrix(use_symmetric=True)
    K_cpu = m_cpu.kernel_matrix()
    eigs_cpu = np.sort(np.linalg.eigvalsh(A_cpu))[::-1]

    # TorchDR PHATE
    m_tdr = PHATEModule(n_components=2, random_state=42, knn=10, t=5, backend="torchdr", device="cpu")
    emb_tdr = m_tdr.fit_transform(x)
    A_tdr = m_tdr.affinity_matrix(use_symmetric=True)
    K_tdr = m_tdr.kernel_matrix()
    eigs_tdr = np.sort(np.linalg.eigvalsh(A_tdr))[::-1]

    # --- Structural invariants (must hold for both) ---

    # Shape
    assert emb_cpu.shape == emb_tdr.shape == (80, 2)
    assert A_cpu.shape == A_tdr.shape == (80, 80)
    assert K_cpu.shape == K_tdr.shape == (80, 80)

    # Symmetry
    assert np.allclose(A_cpu, A_cpu.T, atol=1e-6), "CPU affinity not symmetric"
    assert np.allclose(A_tdr, A_tdr.T, atol=1e-6), "TorchDR affinity not symmetric"

    # Non-negative
    assert A_cpu.min() >= -1e-7, f"CPU affinity has negative values: {A_cpu.min()}"
    assert A_tdr.min() >= -1e-7, f"TorchDR affinity has negative values: {A_tdr.min()}"

    # Leading eigenvalue == 1.0 (within tolerance)
    assert abs(eigs_cpu[0] - 1.0) < 1e-4, f"CPU leading eigenvalue: {eigs_cpu[0]}"
    assert abs(eigs_tdr[0] - 1.0) < 1e-4, f"TorchDR leading eigenvalue: {eigs_tdr[0]}"

    # All eigenvalues in [-1, 1] — the symmetric diffusion operator
    # D^{-1/2} K D^{-1/2} can have negative eigenvalues
    assert eigs_cpu.min() >= -1.01, f"CPU eigenvalue below -1: {eigs_cpu.min()}"
    assert eigs_tdr.min() >= -1.01, f"TorchDR eigenvalue below -1: {eigs_tdr.min()}"
    assert eigs_cpu.max() <= 1.01, f"CPU eigenvalue above 1: {eigs_cpu.max()}"
    assert eigs_tdr.max() <= 1.01, f"TorchDR eigenvalue above 1: {eigs_tdr.max()}"

    # Kernel max == 1.0 (self-affinity)
    assert abs(K_cpu.max() - 1.0) < 1e-4, f"CPU kernel max: {K_cpu.max()}"
    assert abs(K_tdr.max() - 1.0) < 1e-4, f"TorchDR kernel max: {K_tdr.max()}"

    # Spectral gap ratio should be positive for both
    sgr_cpu = eigs_cpu[0] / max(eigs_cpu[1], 1e-10)
    sgr_tdr = eigs_tdr[0] / max(eigs_tdr[1], 1e-10)
    assert sgr_cpu > 1.0, f"CPU spectral gap ratio <= 1: {sgr_cpu}"
    assert sgr_tdr > 1.0, f"TorchDR spectral gap ratio <= 1: {sgr_tdr}"

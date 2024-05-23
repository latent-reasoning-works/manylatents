import numpy as np
from scipy.linalg import solve_banded

def make_fp_explicit(dx=0.1, dt=0.05, sigma=0.5, k=5.0, time_steps=100, N=10, random_seed=42):

    np.random.seed(random_seed)

    x = np.arange(-5, 5, dx)
    all_samples = []
    all_ts = []

    # Initial condition (e.g., Delta function at zero)
    p = np.zeros_like(x)
    p[int(len(x) / 2)] = 1 / dx  # Normalize to form a probability density

    # Time evolution
    for t in range(time_steps):
        dpdx = np.gradient(p, dx)
        d2pdx2 = np.gradient(dpdx, dx)
        drift = -k * x * p
        diffusion = sigma**2 * d2pdx2
        p += dt * (np.gradient(drift, dx) + diffusion)
        p = np.clip(p, 0, None)  # Ensure no negative probabilities
        p /= p.sum() * dx  # Normalize to maintain total probability

        samples = np.random.choice(x, size=N, p=p/p.sum())
        all_samples.append(samples)
        all_ts += [t] * N  # Store time for each sample

    # Convert list of arrays to a single array for analysis
    all_samples = np.hstack(all_samples)
    return p, all_samples, all_ts


def make_fp_implicit(dx=0.1, dt=0.05, sigma=0.5, k=5.0, time_steps=100, N=10, random_seed=42):
    #sigma = Diffusion coefficient
    #k = Drift coefficient
    
    np.random.seed(random_seed)

    x = np.arange(-5, 5, dx)
    all_samples = []
    all_ts = []

    # Initial condition (e.g., Delta function at zero)
    p = np.zeros_like(x)
    p[int(len(x) / 2)] = 1 / dx  # Normalize to form a probability density

    # Coefficients for the tridiagonal matrix for the implicit method
    main_diag = -(2 * sigma**2 / dx**2 + 1 / dt) * np.ones_like(x)
    upper_diag = (sigma**2 / dx**2 - k * dx / 2) * np.ones_like(x[:-1])
    lower_diag = (sigma**2 / dx**2 + k * dx / 2) * np.ones_like(x[:-1])

    # Pad the upper and lower diagonals to match the main diagonal length
    upper_diag = np.append(upper_diag, 0)  # Append zero to the end
    lower_diag = np.insert(lower_diag, 0, 0)  # Insert zero at the beginning

    # Combine into a banded matrix form for the solver
    ab = np.vstack((upper_diag, main_diag, lower_diag))

    # Time evolution
    for t in range(time_steps):
        # Right-hand side of the equation
        rhs = -p / dt
        # Solve the linear system
        p = solve_banded((1, 1), ab, rhs)
        p /= p.sum() * dx  # Normalize to maintain total probability

        samples = np.random.choice(x, size=N, p=p / p.sum())
        all_samples.append(samples)
        all_ts += [t] * N  # Store time for each sample

    # Convert list of arrays to a single array for analysis
    all_samples = np.hstack(all_samples)
    return p, all_samples, all_ts

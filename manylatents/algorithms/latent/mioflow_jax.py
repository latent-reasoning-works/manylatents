"""JAX MIOFlow LatentModule -- trajectory inference via Neural ODE.

Uses diffrax for ODE integration, optax for optimization, and ott-jax
for optimal transport. All JAX deps are lazy-imported.

Reference: Huguet et al., arXiv:2206.14928 (2022)
"""

import logging
from typing import Any

import numpy as np
from torch import Tensor

from manylatents.algorithms.latent.latent_module_base import LatentModule

logger = logging.getLogger(__name__)


class MIOFlowJAX(LatentModule):
    """JAX-based MIOFlow for trajectory inference.

    Learns a time-dependent velocity field dx/dt = f(t, x) using Neural ODE
    with optimal transport losses. Training is hidden inside fit().

    Args:
        n_components: Not used (output dim = input dim for MIOFlow).
        init_seed: Random seed.
        hidden_dim: Width of hidden layers in the velocity MLP.
        n_epochs: Number of training epochs.
        learning_rate: Optax Adam learning rate.
        lambda_ot: Weight for OT loss.
        lambda_energy: Weight for energy regularisation.
        energy_time_steps: Sub-steps for energy loss.
        sample_size: Points sampled per time step (None = all).
        n_trajectories: Trajectories to generate after training.
        n_bins: Time bins for trajectory integration.
    """

    def __init__(
        self,
        n_components: int = 2,
        init_seed: int = 42,
        backend: str | None = None,
        device: str | None = None,
        neighborhood_size: int | None = None,
        # MIOFlow params
        hidden_dim: int = 64,
        n_epochs: int = 100,
        learning_rate: float = 1e-3,
        lambda_ot: float = 1.0,
        lambda_energy: float = 0.01,
        energy_time_steps: int = 10,
        sample_size: int | None = None,
        n_trajectories: int = 100,
        n_bins: int = 100,
        **kwargs,
    ):
        super().__init__(
            n_components=n_components,
            init_seed=init_seed,
            backend=backend,
            device=device,
            neighborhood_size=neighborhood_size,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.lambda_ot = lambda_ot
        self.lambda_energy = lambda_energy
        self.energy_time_steps = energy_time_steps
        self.sample_size = sample_size
        self.n_trajectories = n_trajectories
        self.n_bins = n_bins

        self._params: Any = None
        self._input_dim: int | None = None
        self._time_range: tuple[float, float] = (0.0, 1.0)
        self._trajectories: np.ndarray | None = None

    @staticmethod
    def _make_mlp_params(key, input_dim: int, hidden_dim: int):
        """Initialize MLP parameters: [input_dim+1] -> hidden -> hidden -> [input_dim]."""
        import jax
        import jax.numpy as jnp

        keys = jax.random.split(key, 3)
        scale = 0.1
        params = {
            "w1": jax.random.normal(keys[0], (input_dim + 1, hidden_dim)) * scale,
            "b1": jnp.zeros(hidden_dim),
            "w2": jax.random.normal(keys[1], (hidden_dim, hidden_dim)) * scale,
            "b2": jnp.zeros(hidden_dim),
            "w3": jax.random.normal(keys[2], (hidden_dim, input_dim)) * scale,
            "b3": jnp.zeros(input_dim),
        }
        return params

    @staticmethod
    def _mlp_forward(params, t, x):
        """Forward pass: f(t, x) velocity field.

        Args:
            params: MLP parameter dict.
            t: Scalar time value.
            x: (N, D) batch of positions.

        Returns:
            (N, D) velocity vectors.
        """
        import jax.nn as jnn
        import jax.numpy as jnp

        t_scalar = jnp.asarray(t, dtype=jnp.float32).reshape(())
        t_expanded = jnp.broadcast_to(t_scalar, (x.shape[0],)).reshape(-1, 1)
        h = jnp.concatenate([t_expanded, x], axis=-1)
        h = jnn.silu(h @ params["w1"] + params["b1"])
        h = jnn.silu(h @ params["w2"] + params["b2"])
        return h @ params["w3"] + params["b3"]

    @staticmethod
    def _ode_fn(params):
        """Return a diffrax-compatible vector field function."""

        def vf(t, y, args):
            return MIOFlowJAX._mlp_forward(params, t, y)

        return vf

    @staticmethod
    def _integrate(params, x0, t0, t1, n_steps=10):
        """Integrate ODE from t0 to t1, returning the endpoint.

        Args:
            params: MLP parameter dict.
            x0: (N, D) initial positions.
            t0: Start time (float).
            t1: End time (float).
            n_steps: Number of solver steps hint.

        Returns:
            (N, D) positions at time t1.
        """
        import diffrax
        import jax.numpy as jnp

        t0 = jnp.asarray(t0, dtype=jnp.float32)
        t1 = jnp.asarray(t1, dtype=jnp.float32)

        term = diffrax.ODETerm(MIOFlowJAX._ode_fn(params))
        solver = diffrax.Dopri5()
        dt0 = (t1 - t0) / n_steps
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=x0,
            max_steps=256,
        )
        return sol.ys[0]  # endpoint (the last saved state)

    @staticmethod
    def _integrate_trajectory(params, x0, t_bins):
        """Integrate ODE returning states at all time bins.

        Args:
            params: MLP parameter dict.
            x0: (N, D) initial positions.
            t_bins: 1D array of time values to save at.

        Returns:
            (n_bins, N, D) array of positions at each time bin.
        """
        import diffrax
        import jax.numpy as jnp

        t_bins = jnp.asarray(t_bins, dtype=jnp.float32)
        term = diffrax.ODETerm(MIOFlowJAX._ode_fn(params))
        solver = diffrax.Dopri5()
        dt0 = (t_bins[-1] - t_bins[0]) / len(t_bins)
        saveat = diffrax.SaveAt(ts=t_bins)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_bins[0],
            t1=t_bins[-1],
            dt0=dt0,
            y0=x0,
            saveat=saveat,
            max_steps=4096,
        )
        return sol.ys  # (n_bins, N, D)

    @staticmethod
    def _ot_loss_simple(source, target):
        """Simplified OT loss: mean minimum squared distance (fast fallback).

        For each source point, finds the closest target point and averages
        the squared distances. This is an asymmetric nearest-neighbor proxy
        for OT cost that avoids ott-jax API complexity.

        Args:
            source: (N, D) predicted positions.
            target: (M, D) target positions.

        Returns:
            Scalar loss value.
        """
        import jax.numpy as jnp

        cost_matrix = jnp.sum(
            (source[:, None] - target[None, :]) ** 2, axis=-1
        )
        return jnp.mean(jnp.min(cost_matrix, axis=1))

    @staticmethod
    def _energy_loss(params, x0, t_seq):
        """Penalize large velocity magnitudes along the trajectory.

        Uses Euler steps to approximate the trajectory and accumulates
        squared velocity norms.

        Args:
            params: MLP parameter dict.
            x0: (N, D) initial positions.
            t_seq: 1D array of time steps for evaluation.

        Returns:
            Scalar energy regularisation loss.
        """
        import jax.numpy as jnp

        total = jnp.float32(0.0)
        count = 0
        x = x0
        for i in range(len(t_seq) - 1):
            dx = MIOFlowJAX._mlp_forward(params, t_seq[i], x)
            total = total + jnp.sum(dx**2)
            count += x.shape[0]
            dt = t_seq[i + 1] - t_seq[i]
            x = x + dx * dt  # Euler step for energy computation
        return total / jnp.maximum(jnp.float32(count), jnp.float32(1.0))

    def fit(self, x: Tensor, y: Tensor | None = None) -> None:
        """Train the JAX ODE model on time-labeled data.

        Args:
            x: Input data (N, D) as torch Tensor or numpy array.
            y: Time labels (N,) -- required for MIOFlow.

        Raises:
            ValueError: If y is None (time labels are required).
        """
        import jax
        import jax.numpy as jnp
        import optax

        if y is None:
            raise ValueError("MIOFlowJAX requires time labels (y). Pass y to fit().")

        # Convert to numpy
        x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
        x_jax = jnp.array(x_np, dtype=jnp.float32)
        y_jax = jnp.array(y_np, dtype=jnp.float32)

        self._input_dim = x_np.shape[1]

        # Group by time
        unique_times = np.unique(np.asarray(y_jax))
        unique_times = np.sort(unique_times)
        time_groups = []
        for t in unique_times:
            mask = np.asarray(y_jax) == t
            time_groups.append((jnp.array(x_np[mask], dtype=jnp.float32), float(t)))

        self._time_range = (float(unique_times[0]), float(unique_times[-1]))

        # Initialize params
        key = jax.random.PRNGKey(self.init_seed)
        self._params = self._make_mlp_params(key, self._input_dim, self.hidden_dim)

        # Optimizer
        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(self._params)

        # Build the training step. Since JAX JIT doesn't easily handle
        # variable-length Python lists, we use a simple non-jitted loop
        # over time intervals inside the loss function, and let JAX trace
        # the per-interval computation.
        def train_step(params, opt_state):
            def loss_fn(params):
                total = jnp.float32(0.0)

                for i in range(len(time_groups) - 1):
                    x_start = time_groups[i][0]
                    x_end = time_groups[i + 1][0]
                    t_start = time_groups[i][1]
                    t_end = time_groups[i + 1][1]

                    if self.sample_size is not None:
                        n = min(
                            x_start.shape[0], x_end.shape[0], self.sample_size
                        )
                        x_start = x_start[:n]
                        x_end = x_end[:n]

                    # Integrate from t_start to t_end
                    x_pred = MIOFlowJAX._integrate(
                        params, x_start, t_start, t_end
                    )

                    # OT loss (simplified nearest-neighbor)
                    if self.lambda_ot > 0:
                        n_match = min(x_pred.shape[0], x_end.shape[0])
                        ot_cost = MIOFlowJAX._ot_loss_simple(
                            x_pred[:n_match], x_end[:n_match]
                        )
                        total = total + self.lambda_ot * ot_cost

                # Energy regularisation
                if self.lambda_energy > 0:
                    t_seq = jnp.linspace(
                        time_groups[0][1],
                        time_groups[-1][1],
                        self.energy_time_steps,
                    )
                    total = total + self.lambda_energy * MIOFlowJAX._energy_loss(
                        params, time_groups[0][0], t_seq
                    )

                return total

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        # Training loop
        for epoch in range(self.n_epochs):
            self._params, opt_state, loss = train_step(self._params, opt_state)
            if epoch % max(1, self.n_epochs // 10) == 0:
                logger.info(f"JAX MIOFlow epoch {epoch}: loss={float(loss):.6f}")

        # Generate trajectories
        self._generate_trajectories(time_groups)

        self._is_fitted = True
        logger.info("JAX MIOFlow fitting completed.")

    def _generate_trajectories(self, time_groups):
        """Generate trajectories after training.

        Integrates n_trajectories paths from the initial time point
        through all n_bins time bins and stores in self._trajectories.

        Args:
            time_groups: List of (data_jax, time_float) tuples sorted by time.
        """
        import jax.numpy as jnp

        x0 = time_groups[0][0]
        n = min(self.n_trajectories, x0.shape[0])
        x0_sample = x0[:n]

        t_bins = jnp.linspace(self._time_range[0], self._time_range[1], self.n_bins)

        traj = self._integrate_trajectory(self._params, x0_sample, t_bins)
        self._trajectories = np.asarray(traj)
        logger.info(f"JAX trajectories generated: shape={self._trajectories.shape}")

    def transform(self, x: Tensor) -> Tensor:
        """Integrate x from t_min to t_max, return endpoint positions.

        Args:
            x: (N, D) input data as torch Tensor or numpy array.

        Returns:
            (N, D) torch Tensor of positions at t_max.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        import jax.numpy as jnp
        import torch

        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        x_np = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        x_jax = jnp.array(x_np, dtype=jnp.float32)

        result = self._integrate(
            self._params, x_jax, self._time_range[0], self._time_range[1]
        )
        result_np = np.asarray(result)

        return torch.from_numpy(result_np).float()

    @property
    def trajectories(self) -> np.ndarray | None:
        """Full trajectories (n_bins, n_traj, d) if generated."""
        return self._trajectories

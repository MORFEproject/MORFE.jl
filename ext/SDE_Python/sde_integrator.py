import numpy as np

class SDEIntegrator:
    """
    Integrator for Stratonovich or Itô SDEs with diagonal/scalar noise.

    By default, the SDE is interpreted in the Stratonovich sense:
        dX = f(X,t) dt + g(X,t) ◦ dW .

    Methods:
        'heun'         – Stratonovich–Heun (strong 0.5, no derivative needed)
        'euler'        – Euler–Maruyama (can be Stratonovich with conversion, or Itô)
        'milstein'     – Milstein (can be Stratonovich with conversion, or Itô)

    When using Euler or Milstein with the default interpretation='stratonovich',
    the drift is automatically converted to Itô using:
        f_ito = f + 0.5 * dg/dx * g
    where dg/dx is the diagonal of the Jacobian (provided via diffusion_derivative).
    """

    def __init__(self, drift, diffusion, diffusion_derivative=None):
        self.f = drift
        self.g = diffusion
        self.dg = diffusion_derivative

    def _generate_brownian_increments(self, n_steps, d, n_sims, dt, seed):
        rng = np.random.default_rng(seed)
        return np.sqrt(dt) * rng.normal(size=(n_steps, d, n_sims))

    def simulate(self, x0, t_span, n_steps, n_sims=1,
                 method='heun', interpretation='stratonovich', seed=None):
        """
        Parameters
        ----------
        x0 : array_like, shape (d,)
        t_span : tuple (t0, T)
        n_steps : int
        n_sims : int
        method : {'heun', 'euler', 'milstein'}
            Numerical scheme.
        interpretation : {'stratonovich', 'ito'}
            Only used for 'euler' and 'milstein'. If 'stratonovich' (default),
            the provided drift is a Stratonovich drift and is converted to Itô.
            'heun' always uses the Stratonovich interpretation directly.
        seed : int or None

        Returns
        -------
        t : ndarray (n_steps+1,)
        X : ndarray (n_steps+1, d, n_sims) or (n_steps+1, n_sims) if d=1
        """
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 0:
            d = 1
            x0 = x0.reshape(1)
        else:
            d = x0.shape[0]

        t0, T = t_span
        dt = (T - t0) / n_steps
        t = np.linspace(t0, T, n_steps + 1)

        dW = self._generate_brownian_increments(n_steps, d, n_sims, dt, seed)
        X = np.empty((n_steps + 1, d, n_sims))
        X[0] = x0[:, np.newaxis]

        # --- choose stepping function ---
        if method == 'heun':
            scheme = self._step_heun
        elif method in ('euler', 'milstein'):
            if interpretation == 'stratonovich':
                if self.dg is None:
                    raise ValueError(
                        "Stratonovich conversion requires diffusion_derivative. "
                        "Provide it or use method='heun'."
                    )
                if method == 'euler':
                    scheme = self._step_euler_stratonovich
                else:
                    scheme = self._step_milstein_stratonovich
            elif interpretation == 'ito':
                if method == 'euler':
                    scheme = self._step_euler_ito
                else:
                    if self.dg is None:
                        raise ValueError("Milstein for Itô needs diffusion_derivative.")
                    scheme = self._step_milstein_ito
            else:
                raise ValueError(f"Unknown interpretation '{interpretation}'.")
        else:
            raise ValueError(f"Unknown method '{method}'.")

        Xn = X[0]
        for i in range(n_steps):
            Xn = scheme(Xn, t[i], dt, dW[i])
            X[i + 1] = Xn

        if d == 1:
            X = X[:, 0, :]
        return t, X

    # ----- Itô stepping (unchanged) -----
    def _step_euler_ito(self, x, t, dt, dW):
        return self._step_euler_generic(x, t, dt, dW, use_ito_drift=False)

    def _step_milstein_ito(self, x, t, dt, dW):
        return self._step_milstein_generic(x, t, dt, dW, use_ito_drift=False)

    # ----- Stratonovich via Itô conversion -----
    def _step_euler_stratonovich(self, x, t, dt, dW):
        return self._step_euler_generic(x, t, dt, dW, use_ito_drift=True)

    def _step_milstein_stratonovich(self, x, t, dt, dW):
        return self._step_milstein_generic(x, t, dt, dW, use_ito_drift=True)

    # ----- Generic Euler / Milstein with optional drift conversion -----
    def _step_euler_generic(self, x, t, dt, dW, use_ito_drift):
        drift_val = np.zeros_like(x)
        diff_val = np.zeros_like(x)
        for sim in range(x.shape[1]):
            xi = x[:, sim]
            drift_val[:, sim] = self._drift_effective(xi, t, use_ito_drift)
            diff_val[:, sim] = self.g(xi, t)
        return x + drift_val * dt + diff_val * dW

    def _step_milstein_generic(self, x, t, dt, dW, use_ito_drift):
        drift_val = np.zeros_like(x)
        diff_val = np.zeros_like(x)
        diff_der = np.zeros_like(x)
        for sim in range(x.shape[1]):
            xi = x[:, sim]
            drift_val[:, sim] = self._drift_effective(xi, t, use_ito_drift)
            diff_val[:, sim] = self.g(xi, t)
            diff_der[:, sim] = self.dg(xi, t)
        correction = 0.5 * diff_der * diff_val * (dW**2 - dt)
        return x + drift_val * dt + diff_val * dW + correction

    def _drift_effective(self, xi, t, use_ito_drift):
        f_val = self.f(xi, t)
        if use_ito_drift:  # Stratonovich → Itô conversion
            return f_val + 0.5 * self.dg(xi, t) * self.g(xi, t)
        else:
            return f_val

    # ----- Stratonovich–Heun (unchanged) -----
    def _step_heun(self, x, t, dt, dW):
        n_sims = x.shape[1]
        x_pred = np.zeros_like(x)
        for sim in range(n_sims):
            xi = x[:, sim]
            x_pred[:, sim] = xi + self.f(xi, t) * dt + self.g(xi, t) * dW[:, sim]
        t_next = t + dt
        drift_corr = np.zeros_like(x)
        diff_corr = np.zeros_like(x)
        for sim in range(n_sims):
            drift_corr[:, sim] = self.f(x_pred[:, sim], t_next)
            diff_corr[:, sim] = self.g(x_pred[:, sim], t_next)
        return (x
                + 0.5 * (self._eval_f(x, t) + drift_corr) * dt
                + 0.5 * (self._eval_g(x, t) + diff_corr) * dW)

    def _eval_f(self, x, t):
        res = np.zeros_like(x)
        for sim in range(x.shape[1]):
            res[:, sim] = self.f(x[:, sim], t)
        return res

    def _eval_g(self, x, t):
        res = np.zeros_like(x)
        for sim in range(x.shape[1]):
            res[:, sim] = self.g(x[:, sim], t)
        return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ------------------------------------------------------------
    # Oscillator parameters
    # ------------------------------------------------------------
    omega0 = 1.0      # natural frequency
    gamma  = 1.0      # damping
    sigma  = 0.2      # noise intensity

    # Stratonovich drift (same as the physical model)
    def drift(x, t):
        return np.array([x[1], -omega0**2 * x[0] - gamma * x[1] - 0.0 * x[0]**3])

    # Diffusion (only velocity component is noisy)
    def diffusion(x, t):
        return np.array([0.0, sigma * x[1] + 5.0])

    # Derivative of diffusion w.r.t. state (diagonal of Jacobian)
    def diffusion_der(x, t):
        return np.array([0.0, sigma])

    # ------------------------------------------------------------
    # Simulation settings
    # ------------------------------------------------------------
    x0 = np.array([1.0, 0.0])     # start with position = 1, velocity = 0
    t_span = (0.0, 100.0)
    n_steps = 2000
    n_sims = 500
    seed = 2024

    # Create integrator (Stratonovich by default)
    integrator = SDEIntegrator(drift, diffusion, diffusion_der)

    # 1) Heun (native Stratonovich, no derivative needed)
    t, X_heun = integrator.simulate(
        x0, t_span, n_steps, n_sims, method='heun', seed=seed)

    # 2) Euler with Stratonovich conversion (uses diffusion_der)
    t, X_euler = integrator.simulate(
        x0, t_span, n_steps, n_sims, method='euler', seed=seed)

    # Extract components
    x1_heun = X_heun[:, 0, :]    # position
    x2_heun = X_heun[:, 1, :]    # velocity
    x1_euler = X_euler[:, 0, :]
    x2_euler = X_euler[:, 1, :]

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 4a. Sample trajectories of position (first 5 paths)
    ax1 = axes[0, 0]
    for path in range(5):
        ax1.plot(t, x1_heun[:, path], color='C0', alpha=0.8,
                label='Heun (Stratonovich)' if path == 0 else "")
        ax1.plot(t, x1_euler[:, path], color='C1', linestyle='--', alpha=0.8,
                label='Euler (Stratonovich via Itô)' if path == 0 else "")
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Position x1')
    ax1.set_title('Sample trajectories (5 paths)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 4b. Phase portrait (Heun, 5 paths) – corrected scatter
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for path, col in zip(range(5), colors):
        ax2.plot(x1_heun[:, path], x2_heun[:, path], color=col, alpha=0.7)
        ax2.scatter(x1_heun[0, path], x2_heun[0, path], color=col,
                    marker='o', label='start' if path == 0 else "")
        ax2.scatter(x1_heun[-1, path], x2_heun[-1, path], color=col,
                    marker='s', label='end' if path == 0 else "")
    ax2.set_xlabel('Position x1')
    ax2.set_ylabel('Velocity x2')
    ax2.set_title('Phase portrait (Heun, 5 paths)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 4c. Ensemble mean ± 1 std (Heun)
    ax3 = axes[1, 0]
    mean_heun = np.mean(x1_heun, axis=1)
    std_heun  = np.std(x1_heun, axis=1)
    ax3.fill_between(t, mean_heun - std_heun, mean_heun + std_heun,
                    color='C0', alpha=0.2, label='±1 std')
    ax3.plot(t, mean_heun, color='C0', label='Mean (Stratonovich)')
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Position x1')
    ax3.set_title('Ensemble statistics (Heun, 100 paths)')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4d. Comparison of Heun vs. Euler (Stratonovich conversion)
    ax4 = axes[1, 1]
    mean_euler = np.mean(x1_euler, axis=1)
    ax4.plot(t, mean_heun, color='C0', label='Heun')
    ax4.plot(t, mean_euler, color='C1', linestyle='--', label='Euler (Stratonovich via Itô)')
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Mean position')
    ax4.set_title('Both methods agree (same Stratonovich SDE)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
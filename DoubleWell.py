import numpy as np


class DoubleWellPath:
    """
    Discretized Euclidean path integral for a particle in V(q) = lam*(q^2 - a^2)^2.

    Action:  S[q] = sum_i [ (m/2eps)(q_{i+1} - q_i)^2 + eps * V(q_i) ]

    Periodic BC:  q_{N_tau} = q_0   (thermal partition function)
    Dirichlet BC: q_0 = q_L, q_{N_tau-1} = q_R  (transition amplitude)
    """

    def __init__(self, N_tau, beta, lam, a, m=1.0, bc='periodic',
                 mu_bias=0.0, tau_pin=None):
        self.N_tau = N_tau
        self.beta = beta
        self.eps = beta / N_tau
        self.lam = lam
        self.a = a
        self.m = m
        self.bc = bc  # 'periodic' or 'dirichlet'
        self.mu_bias = mu_bias
        self.tau_pin = tau_pin if tau_pin is not None else beta / 2.0
        tau = np.linspace(0, beta, N_tau, endpoint=False)
        self._bias_weights = ((tau - self.tau_pin) / beta)**2

        # Characteristic instanton frequency
        self.omega = 2.0 * a * np.sqrt(2.0 * lam)

        # Classical instanton action (single kink, continuum)
        self.S_inst_continuum = self.omega**3 / (12.0 * self.lam)
        # Initialize in vacuum
        self.q = self.a * np.ones(N_tau)
        self.action = self.get_action()

    # ----------------------------------------------------------------
    #  Potential and derivatives
    # ----------------------------------------------------------------
    def V(self, q):
        return self.lam * (q**2 - self.a**2)**2

    def dV(self, q):
        return 4.0 * self.lam * q * (q**2 - self.a**2)

    def d2V(self, q):
        return 4.0 * self.lam * (3.0 * q**2 - self.a**2)

    # ----------------------------------------------------------------
    #  Action
    # ----------------------------------------------------------------
    def get_action(self, q=None):
        if q is None:
            q = self.q
        eps, m = self.eps, self.m

        if self.bc == 'periodic':
            dq = np.roll(q, -1) - q
        else:
            dq = np.diff(q)

        kinetic = (m / (2.0 * eps)) * np.sum(dq**2)
        potential = eps * np.sum(self.V(q))
        S = kinetic + potential

        if self.mu_bias > 0:
            S += 0.5 * self.mu_bias * eps * np.sum(self._bias_weights * q**2)

        return S
    
    def get_local_action(self, i, q=None):
        """Action terms that depend on q_i (for single-site Metropolis)."""
        
        if q is None:
            q = self.q
        
        eps, m, N = self.eps, self.m, self.N_tau

        # Potential at site i
        S = eps * self.V(q[i])

        if self.bc == 'periodic':
            ip = (i + 1) % N
            im = (i - 1) % N
            S += (m / (2.0 * eps)) * ((q[i] - q[im])**2 + (q[ip] - q[i])**2)
        else:
            # Interior sites have two bonds; boundary sites have one
            if i > 0:
                S += (m / (2.0 * eps)) * (q[i] - q[i - 1])**2
            if i < N - 1:
                S += (m / (2.0 * eps)) * (q[i + 1] - q[i])**2

        return S

    # ----------------------------------------------------------------
    #  Exact gradient and Hessian (for CNN validation)
    # ----------------------------------------------------------------
    def get_gradient(self, q=None):
        if q is None:
            q = self.q
        eps, m, N = self.eps, self.m, self.N_tau
        grad = np.zeros(N)

        if self.bc == 'periodic':
            grad = (m / eps) * (2.0 * q - np.roll(q, 1) - np.roll(q, -1))
        else:
            grad[1:-1] = (m / eps) * (2.0 * q[1:-1] - q[:-2] - q[2:])
            grad[0] = (m / eps) * (q[0] - q[1])
            grad[-1] = (m / eps) * (q[-1] - q[-2])

        grad += eps * self.dV(q)

        if self.mu_bias > 0:
            grad += self.mu_bias * eps * self._bias_weights * q

        return grad

    def get_hessian(self, q=None):
        if q is None:
            q = self.q
        eps, m, N = self.eps, self.m, self.N_tau
        H = np.zeros((N, N))

        if self.bc == 'periodic':
            np.fill_diagonal(H, 2.0 * m / eps + eps * self.d2V(q))
            for i in range(N):
                H[i, (i + 1) % N] = -m / eps
                H[i, (i - 1) % N] = -m / eps
        else:
            np.fill_diagonal(H, 2.0 * m / eps + eps * self.d2V(q))
            H[0, 0] = m / eps + eps * self.d2V(q[0])
            H[-1, -1] = m / eps + eps * self.d2V(q[-1])
            for i in range(N - 1):
                H[i, i + 1] = -m / eps
                H[i + 1, i] = -m / eps

        if self.mu_bias > 0:
            H[np.diag_indices(N)] += self.mu_bias * eps * self._bias_weights

        return H

    # ----------------------------------------------------------------
    #  Sampling
    # ----------------------------------------------------------------
    def metropolis(self, sigma=0.3):
        N = self.N_tau

        # For Dirichlet BC, don't update boundary sites
        if self.bc == 'dirichlet':
            i = np.random.randint(1, N - 1)
        else:
            i = np.random.randint(0, N)

        q_old = self.q[i]
        S_old = self.get_local_action(i)

        self.q[i] += sigma * np.random.randn()
        S_new = self.get_local_action(i)
        dS = S_new - S_old

        if dS > 0 and np.random.rand() >= np.exp(-dS):
            self.q[i] = q_old
            return False

        self.action += dS
        return True

    def sweep(self, sigma=0.3):
        n_sites = self.N_tau if self.bc == 'periodic' else self.N_tau - 2
        n_accepted = sum(self.metropolis(sigma) for _ in range(n_sites))
        return n_accepted / max(n_sites, 1)

    # ----------------------------------------------------------------
    #  Initialization helpers
    # ----------------------------------------------------------------
    def init_vacuum(self, sign=+1):
        """Initialize in one of the two minima: q_i = ±a."""
        self.q = sign * self.a * np.ones(self.N_tau)
        self.action = self.get_action()

    def init_kink(self, tau_0=None):
        tau = np.linspace(0, self.beta, self.N_tau, endpoint=False)
        if tau_0 is None:
            tau_0 = self.beta / 2.0
        self.q = self.a * np.tanh(self.omega * (tau - tau_0) / 2.0)

        if self.bc == 'dirichlet':
            self.q[0] = -self.a
            self.q[-1] = +self.a

        self.action = self.get_action()

    def init_kink_antikink(self, tau_1=None, tau_2=None):
        """
        Initialize with a kink-antikink pair for periodic BC.
            q(tau) ≈ a * [tanh(omega*(tau-tau_1)/2) - tanh(omega*(tau-tau_2)/2) - 1]
        """
        tau = np.linspace(0, self.beta, self.N_tau, endpoint=False)
        if tau_1 is None:
            tau_1 = self.beta / 4.0
        if tau_2 is None:
            tau_2 = 3.0 * self.beta / 4.0
        kink = np.tanh(self.omega * (tau - tau_1) / 2.0)
        antikink = np.tanh(self.omega * (tau - tau_2) / 2.0)
        self.q = self.a * (kink - antikink - 1.0)
        self.action = self.get_action()


    def mean_field_cloud(
                self,
                n_samples,
                max_mode=8,
                uv_sigma=0.0,
                DeltaS_target=1.0,
                DeltaS_max_factor=3.0,
                center=None,
                S_center=None,
                include_center=True,
        ):
        """
        Build a cloud of configurations around a chosen double-well background
        (vacuum or instanton) using low-frequency Euclidean-time modes.

        ...docstring unchanged...
        """

        N = self.N_tau
        beta = self.beta
        tau = np.linspace(0.0, beta, N, endpoint=False)

        # ------------------------------------------------------------
        # 1. Choose center
        # ------------------------------------------------------------
        if center is None:
            if self.bc == 'dirichlet':
                q0 = self.a * np.tanh(self.omega * (tau - beta / 2.0) / 2.0)
                q0[0] = -self.a
                q0[-1] = +self.a
            elif self.bc == 'periodic':
                q0 = self.a * np.ones(N)
            else:
                raise ValueError(f"Unsupported bc='{self.bc}'")
        else:
            q0 = np.array(center, dtype=float).copy()

        if S_center is None:
            S0 = self.get_action(q0)
        else:
            S0 = float(S_center)
        H = self.get_hessian(q0)

        cfgs = []
        actions = []

        if include_center:
            cfgs.append(q0.copy())
            actions.append(S0)

        # ------------------------------------------------------------
        # 2. Build low-frequency mode basis
        # ------------------------------------------------------------
        basis = []

        if self.bc == 'dirichlet':
            tau_full = np.linspace(0.0, beta, N)
            for n in range(1, max_mode + 1):
                u = np.sin(np.pi * n * tau_full / beta)
                u[0] = 0.0
                u[-1] = 0.0
                norm = np.linalg.norm(u)
                if norm > 1e-12:
                    basis.append(u / norm)

        elif self.bc == 'periodic':
            for n in range(1, max_mode + 1):
                uc = np.cos(2.0 * np.pi * n * tau / beta)
                us = np.sin(2.0 * np.pi * n * tau / beta)
                nc = np.linalg.norm(uc)
                ns = np.linalg.norm(us)
                if nc > 1e-12:
                    basis.append(uc / nc)
                if ns > 1e-12:
                    basis.append(us / ns)

        else:
            raise ValueError(f"Unsupported bc='{self.bc}'")

        if len(basis) == 0:
            raise ValueError("No modes constructed. Increase max_mode.")

        B = np.stack(basis, axis=0)
        n_modes = B.shape[0]

        # ------------------------------------------------------------
        # 3. Mode stiffness from exact Hessian at center
        # ------------------------------------------------------------
        tiny = 1e-10
        Gamma = np.array([u @ H @ u for u in B], dtype=float)
        Gamma = np.maximum(Gamma, tiny)

        sigma_mode = np.sqrt(DeltaS_target / (n_modes * Gamma))

        std_dS = DeltaS_target / np.sqrt(n_modes)
        DeltaS_min = max(0.0, DeltaS_target - DeltaS_max_factor * std_dS)
        DeltaS_max = DeltaS_target + DeltaS_max_factor * std_dS

        # ------------------------------------------------------------
        # 4. Sample cloud
        # ------------------------------------------------------------
        collected = 0
        while collected < n_samples:
            coeffs = sigma_mode * np.random.randn(n_modes)
            delta = np.tensordot(coeffs, B, axes=(0, 0))

            if uv_sigma > 0.0:
                delta += uv_sigma * np.random.randn(N)

            q_trial = q0 + delta

            if self.bc == 'dirichlet':
                q_trial[0] = q0[0]
                q_trial[-1] = q0[-1]

            S_trial = self.get_action(q_trial)
            dS = S_trial - S0

            if DeltaS_min <= dS <= DeltaS_max:
                cfgs.append(q_trial.copy())
                actions.append(S_trial)
                collected += 1

        return np.array(cfgs), np.array(actions)
    

    def instanton_cloud_dataset(self, n_centers=20, samples_per_center=50, max_mode=8,
                                uv_sigma=0.0, DeltaS_target=1.0, tau0_list=None):
        
        if tau0_list is None:
            tau0_list = np.linspace(0.2 * self.beta, 0.8 * self.beta, n_centers)

        all_cfgs = []
        all_actions = []

        tau = np.linspace(0.0, self.beta, self.N_tau, endpoint=False)

        for tau0 in tau0_list:
            center = self.a * np.tanh(self.omega * (tau - tau0) / 2.0)
            if self.bc == 'dirichlet':
                center[0] = -self.a
                center[-1] = +self.a

            cfgs, acts = self.mean_field_cloud(
                n_samples=samples_per_center,
                max_mode=max_mode,
                uv_sigma=uv_sigma,
                DeltaS_target=DeltaS_target,
                center=center,
                include_center=True,
            )
            all_cfgs.append(cfgs)
            all_actions.append(acts)

        return np.concatenate(all_cfgs, axis=0), np.concatenate(all_actions, axis=0)
    

    # Add as methods to DoubleWellPath:

    def topological_charge(self, q=None):
        """Q = (q_final - q_initial) / (2a). Meaningful for Dirichlet BC."""
        if q is None:
            q = self.q
        return 0.5 * (q[-1] - q[0]) / self.a

    def find_zero_crossing(self, q=None):
        """Estimate instanton center τ₀ from linear interpolation of q(τ)=0."""
        if q is None:
            q = self.q
        tau = np.linspace(0, self.beta, self.N_tau, endpoint=False)
        for i in range(self.N_tau - 1):
            if q[i] * q[i + 1] < 0:
                frac = -q[i] / (q[i + 1] - q[i])
                return tau[i] + frac * (tau[i + 1] - tau[i])
        return np.nan

    def classical_kink(self, tau_0=None):
        """Return the classical kink profile (not stored in self.q)."""
        tau = np.linspace(0, self.beta, self.N_tau, endpoint=False)
        if tau_0 is None:
            tau_0 = self.beta / 2.0
        q = self.a * np.tanh(self.omega * (tau - tau_0) / 2.0)
        if self.bc == 'dirichlet':
            q[0] = -self.a
            q[-1] = +self.a
        return q
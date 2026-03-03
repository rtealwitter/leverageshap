import numpy as np
from scipy.special import comb as binom
from .sampling import CoalitionSampler
from .helpers import Game

class LeverageSHAPSNGD:
    def __init__(self, n, game, paired_sampling=True, seed=42):
        self.game = game
        self.n = n
        self.paired_sampling = paired_sampling
        self.seed = seed
        self.P = np.eye(n) - np.ones((n, n)) / n

    def _get_batch_schedule(self, num_samples):
        min_batch = 3 *self.n
        if num_samples < min_batch:
            return [num_samples]
            
        T = num_samples // min_batch
        base_batch = num_samples // T
        remainder = num_samples % T
        
        return [base_batch + (1 if i < remainder else 0) for i in range(T)]

    def _sample_coalitions(self, batch_size, random_state):
        sampling_weights = np.ones(self.n - 1)
        sampler = CoalitionSampler(
            n_players=self.n, 
            sampling_weights=sampling_weights, 
            pairing_trick=self.paired_sampling, 
            random_state=random_state
        )
        sampler.sample(batch_size)
        
        Z = sampler.coalitions_matrix
        sizes = np.sum(Z, axis=1)
        probs = sampler.coalitions_probability

        valid = (sizes > 0) & (sizes < self.n)
        return Z[valid], sizes[valid], probs[valid]

    def _compute_sketched_gradient_and_scale(self, Z, sizes, probs, x_t):
        values = self.game(Z)
        v0, v1 = self.game.edge_cases()
        b_t = values - (v1 - v0) * sizes / self.n

        regression_weights = 1 / (binom(self.n, sizes) * sizes * (self.n - sizes))
        kernel_weights = regression_weights / probs

        # Apply preconditioner P to x_t to ensure we evaluate on the centered subspace
        predictions = Z @ (self.P @ x_t)
        residuals = predictions - b_t
        
        # 1. Unscaled Sketched Gradient
        # Using element-wise multiplication (O(m)) to avoid massive diagonal matrices
        g_raw = self.P @ Z.T @ (kernel_weights * residuals)
        
        # 2. Empirical Trace of the Sketched Hessian
        # The trace of z^T P z is k(n-k)/n. We sum this over the batch weighted by kernel_weights.
        trace_H = np.sum(kernel_weights * sizes * (self.n - sizes) / self.n)
        
        # 3. Scale-Invariant Preconditioner
        # Average eigenvalue in the active (n-1) dimensional subspace
        alpha = trace_H / (self.n - 1)
        
        # Dividing by alpha makes this a true, scale-invariant Stochastic Newton step
        g_t = g_raw / alpha
        
        return g_t

    def shap_values(self, num_samples, eta_0=1.0):
        batch_sizes = self._get_batch_schedule(num_samples)
        v0, v1 = self.game.edge_cases()
        
        x_t = np.zeros(self.n)
        x_avg = np.zeros(self.n)
        update_count = 0
        
        for t, batch_size in enumerate(batch_sizes):
            Z, sizes, probs = self._sample_coalitions(batch_size, random_state=self.seed + t)
            
            if len(sizes) == 0:
                continue
                
            g_t = self._compute_sketched_gradient_and_scale(Z, sizes, probs, x_t)
            
            update_count += 1
            eta_t = eta_0 #/ np.sqrt(update_count)
            
            # The gradient is now optimally scaled, taking a perfect Newton step magnitude
            x_t = x_t - eta_t * g_t
            
            # Polyak-Ruppert Averaging
            x_avg = (x_avg * (update_count - 1) + x_t) / update_count

        return x_t + (v1 - v0) / self.n


def leverage_shap_sngd(baseline, explicand, model, num_samples, subtract_mobius1=False):
    game = Game(model, baseline, explicand, subtract_mobius1=subtract_mobius1)
    n = baseline.shape[1]
    estimator = LeverageSHAPSNGD(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)
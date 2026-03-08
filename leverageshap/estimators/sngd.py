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

    def _sample_valid_coalitions(self, num_samples):
        """Samples all coalitions upfront and filters valid subsets."""
        sampling_weights = np.ones(self.n - 1)
        sampler = CoalitionSampler(
            n_players=self.n, 
            sampling_weights=sampling_weights, 
            pairing_trick=self.paired_sampling, 
            random_state=self.seed
        )
        sampler.sample(num_samples)
        
        Z = sampler.coalitions_matrix
        sizes = np.sum(Z, axis=1)
        probs = sampler.coalitions_probability

        valid = (sizes > 0) & (sizes < self.n)
        return Z[valid], sizes[valid], probs[valid]

    def _get_batch_schedule(self, num_valid_samples):
        """Partitions the total valid samples into dynamic batch sizes."""
        min_batch = 3 * self.n
        if num_valid_samples < min_batch:
            return [num_valid_samples]
            
        T = num_valid_samples // min_batch
        base_batch = num_valid_samples // T
        remainder = num_valid_samples % T
        
        return [base_batch + (1 if i < remainder else 0) for i in range(T)]

    def shap_values(self, num_samples, eta_0=1.0):
        # 1. Sample all data upfront
        Z, sizes, probs = self._sample_valid_coalitions(num_samples)
        num_valid = len(sizes)

        v0, v1 = self.game.edge_cases()
        
        if num_valid == 0:
            return np.zeros(self.n) + (v1 - v0) / self.n

        # 2. Precompute all targets and weights upfront
        values = self.game(Z)
        b = values - (v1 - v0) * sizes / self.n
        
        regression_weights = 1 / (binom(self.n, sizes) * sizes * (self.n - sizes))
        kernel_weights = regression_weights / probs

        # 3. Generate batch partition schedule
        batch_sizes = self._get_batch_schedule(num_valid)
        
        x_t = np.zeros(self.n)
        x_avg = np.zeros(self.n)
        update_count = 0
        
        # 4. Iteration Loop
        start_idx = 0
        for batch_size in batch_sizes:
            end_idx = start_idx + batch_size
            
            # Slice the precomputed arrays for the current batch
            Z_batch = Z[start_idx:end_idx]
            sizes_batch = sizes[start_idx:end_idx]
            b_batch = b[start_idx:end_idx]
            kw_batch = kernel_weights[start_idx:end_idx]
            
            start_idx = end_idx  # Update cursor for the next iteration
            
            if len(sizes_batch) == 0:
                continue

            # Compute Unscaled Sketched Gradient
            predictions = Z_batch @ (self.P @ x_t)
            residuals = predictions - b_batch
            g_raw = self.P @ Z_batch.T @ (kw_batch * residuals)
            
            # Compute Scale-Invariant Preconditioner
            trace_H = np.sum(kw_batch * sizes_batch * (self.n - sizes_batch) / self.n)
            
            if trace_H == 0:
                continue
                
            alpha = trace_H / (self.n - 1)
            g_t = g_raw / alpha
            
            # Update Step
            update_count += 1
            eta_t = eta_0 / np.sqrt(update_count)
            
            x_t = x_t - eta_t * g_t
            
            # Polyak-Ruppert Averaging
            x_avg = (x_avg * (update_count - 1) + x_t) / update_count

        return x_avg + (v1 - v0) / self.n

def leverage_shap_sngd(baseline, explicand, model, num_samples, subtract_mobius1=False):
    game = Game(model, baseline, explicand, subtract_mobius1=subtract_mobius1)
    n = baseline.shape[1]
    estimator = LeverageSHAPSNGD(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)
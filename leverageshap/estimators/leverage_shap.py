import numpy as np
from scipy.special import comb as binom

from .sampling import CoalitionSampler
from ..utils import Game



class LeverageSHAP:
    def __init__(self, n, game, paired_sampling=True, random_state=None):
        self.game = game
        self.n = n
        self.paired_sampling = paired_sampling
        # None -> fresh randomness on every call (so repeated runs are independent).
        self.random_state = random_state
    
    def shap_values(self, num_samples):
        # Sample
        #self.sample()
        # A = Z P
        # y = v(z) - v0
        # b = y - Z1 (v1 - v0) / n    
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n
        # Algorithm 1 assumes m >= n; with fewer samples the projected system
        # is rank deficient and lstsq returns the min-norm solution.  We keep a
        # hard floor of 6 so the paired sampler always has rows to work with.
        if num_samples < 6:
            print(f'Warning: num_samples={num_samples} < 6 for Leverage SHAP (n={self.n}); using 6 samples instead.')
            num_samples = 6

        sampling_weights = np.ones(self.n-1)

        sampler = CoalitionSampler(n_players=self.n, sampling_weights=sampling_weights, pairing_trick=self.paired_sampling, random_state=self.random_state)
        sampler.sample(num_samples)
        coalition_matrix = sampler._sampled_coalitions_matrix
        coalition_sizes = np.sum(coalition_matrix, axis=1)
        sampling_probs = sampler._sampled_coalitions_probability

        # Filter out empty and full coalitions
        filtered_indices = np.where((coalition_sizes > 0) & (coalition_sizes < self.n))[0]
        coalition_matrix = coalition_matrix[filtered_indices]
        coalition_sizes = coalition_sizes[filtered_indices]
        sampling_probs = sampling_probs[filtered_indices]

        values = self.game.value(coalition_matrix)
        
        v0, v1 = self.game.edge_cases()
        # b = (v(z) - v0) - (v1 - v0) |z| / n.  The -v0 term cancels under paired
        # sampling (z + z_bar = 1 is annihilated by P) but matters without it.
        values_adjusted = (values - v0) - (v1 - v0) * coalition_sizes / self.n
        regression_weights = 1 / (binom(self.n, coalition_sizes) * coalition_sizes * (self.n - coalition_sizes))
        kernel_weights = regression_weights / sampling_probs

        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        Atb = P @ coalition_matrix.T @ np.diag(kernel_weights) @ values_adjusted
        AtA = P @ coalition_matrix.T @ np.diag(kernel_weights) @ coalition_matrix @ P

        # AtA is always singular (P has rank n-1); lstsq returns the min-norm
        # solution, which lies in the range of P and is exactly Algorithm 1's
        # constrained least-squares solution.  No ridge term is needed.
        AtA_inv_Atb = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        
        return AtA_inv_Atb + (v1 - v0) / self.n

def leverage_shap(baseline, explicand, model, num_samples, random_state=None):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = LeverageSHAP(n, game, paired_sampling=True, random_state=random_state)
    return estimator.shap_values(num_samples)

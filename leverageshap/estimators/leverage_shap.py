import numpy as np
from .sampling import CoalitionSampler
from .helpers import Game
from scipy.special import comb as binom

class LeverageSHAP:
    def __init__(self, n, game, paired_sampling=True):
        self.game = game
        self.n = n
        self.paired_sampling = paired_sampling
    
    def shap_values(self, num_samples):
        # Sample
        #self.sample()
        # A = Z P
        # y = v(z) - v0
        # b = y - Z1 (v1 - v0) / n    
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n
        if num_samples < 6:
            print('Number of samples too small, setting to 6')
            num_samples = 6

        sampling_weights = np.ones(self.n-1)

        sampler = CoalitionSampler(n_players=self.n, sampling_weights=sampling_weights, pairing_trick=self.paired_sampling)
        sampler.sample(num_samples)
        coalition_matrix = sampler.coalitions_matrix
        coalition_sizes = np.sum(coalition_matrix, axis=1)
        sampling_probs = sampler.sampling_probabilities

        # Filter out empty and full coalitions
        filtered_indices = np.where((coalition_sizes > 0) & (coalition_sizes < self.n))[0]
        coalition_matrix = coalition_matrix[filtered_indices]
        coalition_sizes = coalition_sizes[filtered_indices]
        sampling_probs = sampling_probs[filtered_indices]

        values = self.game(coalition_matrix)
        
        v0, v1 = self.game.edge_cases()
        values_adjusted = values - (v1 - v0) * coalition_sizes/ self.n
        regression_weights = 1 / (binom(self.n, coalition_sizes) * coalition_sizes * (self.n - coalition_sizes))
        kernel_weights = regression_weights / sampling_probs

        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        Atb = P @ coalition_matrix.T @ np.diag(kernel_weights) @ values_adjusted
        AtA = P @ coalition_matrix.T @ np.diag(kernel_weights) @ coalition_matrix @ P

        if np.linalg.cond(AtA) > 1 / np.finfo(AtA.dtype).eps and num_samples <= 3*self.n:
            sqrt_alpha = 1e-3
            AtA = AtA + sqrt_alpha * np.eye(AtA.shape[0])

            yellow_start="\033[33m"
            yellow_end="\033[0m"
            print(f'{yellow_start}Warning:{yellow_end} Singular matrix in Leverage SHAP with num_samples={num_samples} and num_players={self.n}, adding ridge regularization with alpha={sqrt_alpha**2}.')

        AtA_inv_Atb = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        
        return AtA_inv_Atb + (v1 - v0) / self.n

def leverage_shap(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = LeverageSHAP(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)

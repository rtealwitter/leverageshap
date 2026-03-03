import numpy as np
from scipy.special import binom

from .sampling import CoalitionSampler
from ..utils import Game

class LeverageSHAPSNGD:
    def __init__(self, n, game, paired_sampling=True):
        self.game = game
        self.n = n
        self.paired_sampling = paired_sampling
        
    def shap_values(self, num_samples, eta_0=1.0):
        # 1. Dynamically choose T and batch sizes (at least 5*n per iteration)
        min_samples_per_iter = 5 * self.n
        
        if num_samples < min_samples_per_iter:
            print(f'Budget too small for multiple iterations. Using 1 iteration of size {num_samples}.')
            T = 1
            batch_sizes = [num_samples]
        else:
            T = num_samples // min_samples_per_iter
            base_batch = num_samples // T
            remainder = num_samples % T
            # Distribute remainder evenly across the first few batches
            batch_sizes = [base_batch + (1 if i < remainder else 0) for i in range(T)]


        # Replace `exact_AtA` with your analytical exact AtA matrix.
        exact_AtA = 1/self.n * (np.eye(self.n) - 1/self.n * np.ones((self.n, self.n)))

        # Precompute the pseudo-inverse of the exact Hessian.
        # (pinv is required because the centering matrix P makes AtA rank n-1)
        exact_H_inv = np.linalg.pinv(exact_AtA)

        # Initialize parameter vector x_t and the Polyak-Ruppert running average
        x_t = np.zeros(self.n)
        x_avg = np.zeros(self.n)

        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))
        v0, v1 = self.game.edge_cases()

        for t, batch_size in enumerate(batch_sizes):
            # 2. Sample the mini-batch
            sampling_weights = np.ones(self.n-1)
            # Use t to change the random state per iteration
            sampler = CoalitionSampler(n_players=self.n, sampling_weights=sampling_weights, 
                                       pairing_trick=self.paired_sampling, random_state=42+t)
            sampler.sample(batch_size)
            
            coalition_matrix = sampler._sampled_coalitions_matrix
            coalition_sizes = np.sum(coalition_matrix, axis=1)
            sampling_probs = sampler._sampled_coalitions_probability

            # 3. Filter out empty and full coalitions
            filtered_indices = np.where((coalition_sizes > 0) & (coalition_sizes < self.n))[0]
            Z_t = coalition_matrix[filtered_indices]
            sizes_t = coalition_sizes[filtered_indices]
            probs_t = sampling_probs[filtered_indices]

            if len(sizes_t) == 0:
                continue # Skip update if the batch only contained empty/full coalitions

            # 4. Compute game values and targets
            values = self.game.value(Z_t)
            b_t = values - (v1 - v0) * sizes_t / self.n

            # 5. Compute weights (importance sampling scaled)
            regression_weights = 1 / (binom(self.n, sizes_t) * sizes_t * (self.n - sizes_t))
            kernel_weights = regression_weights / probs_t

            # 6. Form sketched gradient components
            # Dividing by batch_size makes these unbiased estimators of the true power-set sums
            sketched_Atb_t = (P @ Z_t.T @ np.diag(kernel_weights) @ b_t) / batch_size
            sketched_AtA_t = (P @ Z_t.T @ np.diag(kernel_weights) @ Z_t @ P) / batch_size

            # 7. Compute Sketched Gradient: g_t = (A_S^T A_S) x_t - A_S^T b_S
            g_t = sketched_AtA_t @ x_t - sketched_Atb_t

            # 8. Preconditioned Update step with learning rate schedule
            # Using 1 / sqrt(t+1) schedule for robust convergence with averaging
            eta_t = eta_0 / np.sqrt(t + 1)
            x_t = x_t - eta_t * (exact_H_inv @ g_t)

            # 9. Polyak-Ruppert Averaging
            x_avg = (x_avg * t + x_t) / (t + 1)

        # 10. Add the baseline component back to the averaged solution
        return x_avg + (v1 - v0) / self.n

def leverage_shap_sngd(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = LeverageSHAPSNGD(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)
import numpy as np
from .sampling import CoalitionSampler
from .helpers import Game
from scipy.special import comb as binom

# Helpers

from typing import Iterable, List, Sequence, Tuple, Optional

def shapley_from_fourier(
    interactions: Sequence[Iterable[int]],
    coeffs: Sequence[float],
    n: int,
) -> List[float]:
    shapley_values = np.zeros(n)
    for S, coeff in zip(interactions, coeffs):
        if len(S) % 2 == 0:
            continue
        for i in S:
            shapley_values[i] += coeff / len(S)
    return shapley_values #* -2

def interactions_to_matrix(interactions: Sequence[Iterable[int]], n: int) -> np.ndarray:
    """
    Build a k×n binary matrix A for k interaction terms over n columns.
    A[j, i] = 1 iff i ∈ interactions[j].
    """
    A = np.zeros((len(interactions), n), dtype=np.uint8)
    for j, T in enumerate(interactions):
        cols = np.fromiter(T, dtype=int)
        if cols.size:
            if (cols < 0).any() or (cols >= n).any():
                raise IndexError(f"Interaction {j} has indices outside 0..{n-1}")
            A[j, cols] = 1
    return A

def fourier_design_matrix(coalition_matrix: np.ndarray,
                   interactions: Sequence[Iterable[int]]) -> np.ndarray:
    """
    Compute F ∈ {−1,+1}^{m×k} with F[S,T] = (-1)^{|S∩T|}.
    Efficiently computed via parity of C @ A^T.
    """
    m, n = coalition_matrix.shape
    A = interactions_to_matrix(interactions, n)      # k×n
    # Integer product then mod 2 to get parity:
    parity = (coalition_matrix.astype(int) @ A.T) % 2   # (m×k) in {0,1}
    F = 1 - 2*parity                                         # map 0→+1, 1→−1
    return F



class NewSHAP:
    def __init__(self, n, game, paired_sampling=True):
        self.game = game
        self.n = n
        self.paired_sampling = paired_sampling
    
    def shap_values(self, num_samples):
        # Solve argmin_{x: <x,1>=v1-v0} (Ax - b)^T W (Ax - b)
        if num_samples < 6:
            print('Number of samples too small, setting to 6')
            num_samples = 6

        sampling_weights = np.ones(self.n)

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

        values = self.game.value(coalition_matrix)
        v0, v1 = self.game.edge_cases()

        # Convert to Fourier
        interactions = [(i,) for i in range(self.n)]
#        interactions += [(i, j, k) for i in range(self.n) for j in range(i+1, self.n) for k in range(j+1, self.n)]

        coalition_matrix = fourier_design_matrix(coalition_matrix, interactions=interactions)
        values = - 2 * values + (v1 - v0)

        row_sum = np.sum(coalition_matrix, axis=1)

        values_adjusted = values - (v1 - v0) * row_sum/ self.n
        regression_weights = 1 / (binom(self.n, coalition_sizes) * coalition_sizes * (self.n - coalition_sizes))
        kernel_weights = regression_weights / sampling_probs

        dim = coalition_matrix.shape[1]
        P = np.eye(dim) - 1/dim

        Atb = P @ coalition_matrix.T @ np.diag(kernel_weights) @ values_adjusted
        AtA = P @ coalition_matrix.T @ np.diag(kernel_weights) @ coalition_matrix @ P

        if np.linalg.cond(AtA) > 1 / np.finfo(AtA.dtype).eps and num_samples <= 3*self.n:
            sqrt_alpha = 1e-3
            AtA = AtA + sqrt_alpha * np.eye(AtA.shape[0])

            yellow_start="\033[33m"
            yellow_end="\033[0m"
            print(f'{yellow_start}Warning:{yellow_end} Singular matrix in Leverage SHAP with num_samples={num_samples} and num_players={self.n}, adding ridge regularization with alpha={sqrt_alpha**2}.')

        AtA_inv_Atb = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        
        coeffs = AtA_inv_Atb + (v1 - v0) / self.n
        #return coeffs

        shap_values = shapley_from_fourier(interactions=interactions, coeffs=coeffs, n=self.n)

        return shap_values

def new_shap(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)
import numpy as np
from .sampling import CoalitionSampler
from .helpers import Game
from scipy.special import comb as binom
from shapiq.game import Game as ShapiqGame

# Helpers

from typing import Iterable, List, Sequence, Tuple, Optional

import shapiq
from sparse_transform.qsft.utils.query import get_bch_decoder
from itertools import chain, combinations

def powerset(iterable):
    s = tuple(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def mobius_to_fourier(mobius_dict):
    """
    Convert Möbius coefficients to Fourier coefficients.
    """
    if not mobius_dict:
        return {}

    unscaled_fourier_dict = {}
    for loc, coef in mobius_dict.items():
        real_coef = np.real(coef) / (2 ** len(loc))   # cardinality, not sum of indices
        for subset in powerset(loc):
            unscaled_fourier_dict[subset] = unscaled_fourier_dict.get(subset, 0.0) + real_coef

    # multiply each entry by (-1)^(|loc|)
    return {
        loc: val * ((-1.0) ** len(loc))
        for loc, val in unscaled_fourier_dict.items()
        if abs(val) > 1e-12
    }

def spex_top_fourier(game, n, t, b):
    spex_approximator = shapiq.SPEX(n=n, index="FSII", max_order = 3)
    spex_approximator.degree_parameter = t
    spex_approximator.query_args["t"] = t
    spex_approximator.decoder_args["source_decoder"] = get_bch_decoder(n, t, "soft")

    budget = int(3 * np.log2(n) * 2**b * t)
    print(f'Budget for SPEX: {budget}')
    mobius_inters = spex_approximator.approximate(game=game, budget=budget).dict_values
    fourier_inters = mobius_to_fourier(mobius_inters)
    fourier_inters_useful = {}
    for loc, coef in fourier_inters.items():
        if len(loc) >= 2 and len(loc) % 2 == 1:
            fourier_inters_useful[loc] = coef
    # Sort by absolute value
    fourier_inters_useful = dict(sorted(fourier_inters_useful.items(), key=lambda item: abs(item[1]), reverse=True))
    print(f'Found {len(fourier_inters_useful)} Fourier interactions: {fourier_inters_useful}.')
    return list(fourier_inters.keys())

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
    return shapley_values * -2

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
    def __init__(self, n, game, interaction_strategy, paired_sampling=True):
        self.game = game
        self.n = n
        assert interaction_strategy in ['SPEX', 'ProxySPEX', 'Guess']
        self.interaction_strategy = interaction_strategy
        self.paired_sampling = paired_sampling
    
    def setupandsolve(self, interactions, sampled_coalitions, values, sampling_probs):
        # Solve argmin_{x: <x,1>=v1-v0} (Ax - b)^T W (Ax - b)
        coalition_sizes = np.sum(sampled_coalitions, axis=1)
        v0, v1 = values[np.where(coalition_sizes == 0)[0][0]], values[np.where(coalition_sizes == self.n)[0][0]]
        # Filter out empty and full coalitions
        filtered_indices = np.where((coalition_sizes > 0) & (coalition_sizes < self.n))[0]
        sampled_coalitions = sampled_coalitions[filtered_indices]
        coalition_sizes = coalition_sizes[filtered_indices]
        sampling_probs = sampling_probs[filtered_indices]
        values = values[filtered_indices]

        coalition_matrix = fourier_design_matrix(sampled_coalitions, interactions=interactions)

        dim = coalition_matrix.shape[1]
        constant = (v1 - v0) / dim
        constant *= -1/2

        row_sum = np.sum(coalition_matrix, axis=1)

        values_adjusted = values - constant * row_sum
        regression_weights = 1 / (binom(self.n, coalition_sizes) * coalition_sizes * (self.n - coalition_sizes))
        kernel_weights = regression_weights / sampling_probs

        
        P = np.eye(dim) - 1/dim

        Atb = P @ coalition_matrix.T @ np.diag(kernel_weights) @ values_adjusted
        AtA = P @ coalition_matrix.T @ np.diag(kernel_weights) @ coalition_matrix @ P

        if np.linalg.cond(AtA) > 1 / np.finfo(AtA.dtype).eps and coalition_matrix.shape[0] <= 5*coalition_matrix.shape[1]:
            sqrt_alpha = 1e-3
            AtA = AtA + sqrt_alpha * np.eye(AtA.shape[0])

            yellow_start="\033[33m"
            yellow_end="\033[0m"
            print(f'{yellow_start}Warning:{yellow_end} Singular matrix in Leverage SHAP with num_samples={sampled_coalitions.shape[0]} and num_players={self.n}, adding ridge regularization with alpha={sqrt_alpha**2}.')

        AtA_inv_Atb = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        
        coeffs = AtA_inv_Atb + constant

        shap_values = shapley_from_fourier(interactions=interactions, coeffs=coeffs, n=self.n)

        return shap_values
    
    def shap_values(self, num_samples):
        if num_samples < 6:
            print('Number of samples too small, setting to 4.')
            num_samples = 4

        interactions = [(i,) for i in range(self.n)]

        if self.interaction_strategy == 'SPEX':
            fourier_inters = spex_top_fourier(self.game, self.n, t=3, b=6)  
            for inter in fourier_inters:
                if len(inter) >= 2 and len(inter) % 2 == 1:
                    interactions += [inter]

        sampling_weights = np.ones(self.n)

        sampler = CoalitionSampler(n_players=self.n, sampling_weights=sampling_weights, pairing_trick=self.paired_sampling)
        sampler.sample(num_samples)
        sampled_coalitions = sampler.coalitions_matrix
        sampling_probs = sampler.sampling_probabilities
        values = self.game(sampled_coalitions)

        shap_values = self.setupandsolve(interactions, sampled_coalitions, values, sampling_probs)

        return shap_values

def new_shap(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='SPEX', paired_sampling=True)
    return estimator.shap_values(num_samples)
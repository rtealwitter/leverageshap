import numpy as np
from .sampling import CoalitionSampler
from .helpers import Game
from scipy.special import comb as binom
import time
from .proxyspex import proxyspex

# Helpers

from typing import Iterable, List, Sequence, Tuple, Optional

import shapiq
from sparse_transform.qsft.utils.query import get_bch_decoder
from itertools import chain, combinations
from sklearn.linear_model import LassoCV

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

def spex_top_fourier(game, n, spex_params):
    spex_approximator = shapiq.SPEX(n=n, index="FSII", max_order = n, top_order=True)
    spex_approximator.degree_parameter = spex_params['t']
    spex_approximator.query_args["t"] = spex_params['t']
    spex_approximator.decoder_args["source_decoder"] = get_bch_decoder(n, spex_params['t'], "soft")
    mobius_inters = spex_approximator.approximate(game=game, budget=spex_params['budget']).dict_values
    fourier_inters = mobius_to_fourier(mobius_inters)
    # Sort by absolute value
    fourier_inters = {k : v for k, v in fourier_inters.items() if len(k) == 3}
    fourier_inters = dict(sorted(fourier_inters.items(), key=lambda item: abs(item[1]), reverse=True))

    # ensure keys are plain Python ints (not numpy types)
    return [tuple(int(x) for x in k) for k in list(fourier_inters.keys())]

def proxy_spex_top_fourier(game, n, top_k, budget):
    # proxyspex_approximator = shapiq.ProxySPEX(n=n, index="FSII", max_order=n)
    # mobius_inters = proxyspex_approximator.approximate(game=game, budget=budget).dict_values
    # print(mobius_inters)
    # # print out the value counts of the key lengths of mobius_inters
    # length_counts = {}
    # for key in mobius_inters.keys():
    #     length = len(key)
    #     length_counts[length] = length_counts.get(length, 0) + 1
    # print("Length counts of mobius_inters:", length_counts)
    # fourier_inters = mobius_to_fourier(mobius_inters)
    # return list(fourier_inters.keys())
    # convert any numpy integer types to Python ints
    return [tuple(int(x) for x in k) for k in proxyspex(game, budget, n, top_k=top_k).keys()]


def lasso_top_fourier(game, n, top_k, samples):
    """Fit Lasso on uniform samples, rank first-order terms by |beta|, then build
    third-order interaction candidates from the top-ranked features and return
    the top_k triplets by summed |beta| score.

    Returns a list of tuples (i,j,k).
    """
    random_state = 0
    if top_k <= 0:
        return []

    if isinstance(samples, int):
        # sample uniformly using the CoalitionSampler
        sampler = CoalitionSampler(
            n_players=n,
            sampling_weights=np.array([binom(n, i) for i in range(n + 1)]),
            pairing_trick=True,
            random_state=random_state,
        )
        sampler.sample(samples)
        X = sampler.coalitions_matrix.astype(float)
        y = game(X)
    else:
        X = samples[0].astype(float)
        y = samples[1]

    # fit Lasso on first-order features
    lasso = LassoCV(cv=5, n_alphas=100, random_state=0).fit(X, y)
    coefs = lasso.coef_
    abs_coefs = np.abs(coefs)

    # choose how many top first-order features to consider
    top_l = min(n, max(10, top_k))
    # ensure there are enough combinations to produce top_k triplets
    from math import comb as int_comb
    while int_comb(top_l, 3) < top_k and top_l < n:
        top_l += 1

    if top_l < 3:
        return []

    ranked_idx = np.argsort(abs_coefs)[::-1]
    top_features = list(ranked_idx[:top_l])

    # form all triplets from top_features and score by sum of |beta|
    triplets = list(combinations(top_features, 3))
    scored = sorted(triplets, key=lambda t: abs_coefs[t[0]] + abs_coefs[t[1]] + abs_coefs[t[2]], reverse=True)

    top_triplets = [tuple(int(x) for x in tuple(sorted(t))) for t in scored[:top_k]]
    return top_triplets

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

def get_spex_params(n, m): 
    # budget = C * log2(n) * 2^b * t >= m /2
    t = 3
    C = 3
    b = int(
        np.log2(m / (2 * C * t * np.log2(n)))
    )
    return {'t': t, 'b': b, 'budget': int(2 * C * t * np.log2(n) * 2**b), 'C': C}

class NewSHAP:
    def __init__(self, n, game, interaction_strategy, paired_sampling=True, top_k=None):
        self.game = game
        self.n = n
        self.interaction_strategy = interaction_strategy
        self.paired_sampling = paired_sampling
        if "ProxySPEX" in interaction_strategy or "LASSO" in interaction_strategy:
            assert top_k is not None, "top_k must be specified for ProxySPEX or LASSO"
        self.top_k = top_k

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

        Atb = P @ coalition_matrix.T * kernel_weights @ values_adjusted
        AtA = P @ coalition_matrix.T * kernel_weights @ coalition_matrix @ P

        if np.linalg.cond(AtA) > 1 / np.finfo(AtA.dtype).eps and coalition_matrix.shape[0] <= 5*coalition_matrix.shape[1]:
            sqrt_alpha = 1e-3
            AtA = AtA + sqrt_alpha * np.eye(AtA.shape[0])

            yellow_start="\033[33m"
            yellow_end="\033[0m"
            print(f'{yellow_start}Warning:{yellow_end} Singular matrix in New SHAP with num_samples={sampled_coalitions.shape[0]} and num_players={self.n}, adding ridge regularization with alpha={sqrt_alpha**2}.')

        AtA_inv_Atb = np.linalg.lstsq(AtA, Atb, rcond=None)[0]
        
        coeffs = AtA_inv_Atb + constant

        shap_values = shapley_from_fourier(interactions=interactions, coeffs=coeffs, n=self.n)

        return shap_values
    
    def shap_values(self, num_samples):
        if num_samples < 6:
            print('Number of samples too small, setting to 4.')
            num_samples = 4

        if self.interaction_strategy == 'SPEX':
            spex_params = get_spex_params(self.n, num_samples)
            try:
                new_interactions = spex_top_fourier(
                    self.game, self.n, spex_params
                )  
                print('Using spex with m=', num_samples)
                num_samples -= spex_params['budget']
            except Exception as e:
                new_interactions = []
        if self.interaction_strategy == 'ProxySPEX uniform':
            proxyspex_budget = num_samples // 2
            try:
                new_interactions = proxy_spex_top_fourier(self.game, self.n, self.top_k, samples=proxyspex_budget)
                num_samples -= proxyspex_budget
            except Exception as e:
                new_interactions = []
        if self.interaction_strategy == 'LASSO uniform':
            lasso_budget = num_samples // 4
            try:
                new_interactions = lasso_top_fourier(
                    self.game, self.n, self.top_k, samples=lasso_budget
                )
                num_samples -= lasso_budget
            except Exception as e:
                new_interactions = []

        sampler = CoalitionSampler(n_players=self.n, sampling_weights=np.ones(self.n-1), pairing_trick=self.paired_sampling)
        sampler.sample(num_samples)
        sampled_coalitions = sampler.coalitions_matrix
        sampling_probs = sampler.sampling_probabilities
        values = self.game(sampled_coalitions)

        interactions = [(i,) for i in range(self.n)]

        if self.interaction_strategy == 'Sample':
            shap_values_guess = self.setupandsolve(interactions, sampled_coalitions, values, sampling_probs)
            dist = np.abs(shap_values_guess) / np.sum(np.abs(shap_values_guess))
            # Choose k so that m >= C * (n + k)
            new_interactions = set()
            k = max(0, (num_samples // 10) - self.n)
            for _ in range(k):                
                inter = tuple(np.random.choice(self.n, size=3, replace=False, p=dist))
                inter = tuple(sorted(inter))
                new_interactions.add(inter)
                if len(new_interactions) >= k: break
        if self.interaction_strategy == 'None':
            new_interactions = []
        if self.interaction_strategy == 'ProxySPEX kernel':
            # use ProxySPEX on kernel sampled coalitions
            try:
                new_interactions = proxy_spex_top_fourier(self.game, self.n, self.top_k, samples=(sampled_coalitions, values))
            except Exception as e:
                new_interactions = []
        if self.interaction_strategy == 'LASSO kernel':
            # use LASSO on kernel sampled coalitions
            try:
                new_interactions = lasso_top_fourier(self.game, self.n, self.top_k, samples=(sampled_coalitions, values))
            except Exception as e:
                new_interactions = []
        if self.interaction_strategy == 'Deterministic':
            print('Using deterministic with m=', num_samples)
            new_interactions = []
            shap_values_guess = self.setupandsolve(interactions, sampled_coalitions, values, sampling_probs)
            sorted_indices = np.argsort(-np.abs(shap_values_guess))
            # Choose s so that m >= C * (n + s choose 3)
            # s choose 3 = m / C - n
            # s^3 / 6 approx = m / C - n
            # s= (6 * (m / C - n))^(1/3)
            C = 5
            s = int((6 * max(0, num_samples / C - self.n)) ** (1/3))
            s = min(s, self.n)
            for idx in sorted_indices[:s]:
                for jdx in sorted_indices[:s]:
                    for kdx in sorted_indices[:s]:
                        if idx < jdx < kdx:
                            new_interactions.append((idx, jdx, kdx))
        new_interactions = [inter for inter in new_interactions if len(inter) >= 2 and len(inter) % 2 == 1]
        print(self.interaction_strategy, 'with', len(new_interactions), 'new interactions:', new_interactions) 
        interactions += new_interactions
        shap_values = self.setupandsolve(interactions, sampled_coalitions, values, sampling_probs)
        return shap_values

def spex_shap(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='SPEX', paired_sampling=True)
    return estimator.shap_values(num_samples)

def proxyspex_uniform_shap_paired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX uniform', paired_sampling=True, top_k=1*n)
    return estimator.shap_values(num_samples)

def proxyspex_uniform_shap_paired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX uniform', paired_sampling=True, top_k=2*n)
    return estimator.shap_values(num_samples)

def proxyspex_kernel_shap_paired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX kernel', paired_sampling=True, top_k=1*n)
    return estimator.shap_values(num_samples)

def proxyspex_kernel_shap_paired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX kernel', paired_sampling=True, top_k=2*n)
    return estimator.shap_values(num_samples)

def lasso_uniform_shap_paired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO uniform', paired_sampling=True, top_k=1*n)
    return estimator.shap_values(num_samples)

def lasso_uniform_shap_paired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO uniform', paired_sampling=True, top_k=2*n)
    return estimator.shap_values(num_samples)

def lasso_kernel_shap_paired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO kernel', paired_sampling=True, top_k=1*n)
    return estimator.shap_values(num_samples)

def lasso_kernel_shap_paired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO kernel', paired_sampling=True, top_k=2*n)
    return estimator.shap_values(num_samples)

def proxyspex_uniform_shap_unpaired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX uniform', paired_sampling=False, top_k=1*n)
    return estimator.shap_values(num_samples)

def proxyspex_uniform_shap_unpaired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX uniform', paired_sampling=False, top_k=2*n)
    return estimator.shap_values(num_samples)

def proxyspex_kernel_shap_unpaired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX kernel', paired_sampling=False, top_k=1*n)
    return estimator.shap_values(num_samples)

def proxyspex_kernel_shap_unpaired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='ProxySPEX kernel', paired_sampling=False, top_k=2*n)
    return estimator.shap_values(num_samples)

def lasso_uniform_shap_unpaired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO uniform', paired_sampling=False, top_k=1*n)
    return estimator.shap_values(num_samples)

def lasso_uniform_shap_unpaired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO uniform', paired_sampling=False, top_k=2*n)
    return estimator.shap_values(num_samples)

def lasso_kernel_shap_unpaired_1n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO kernel', paired_sampling=False, top_k=1*n)
    return estimator.shap_values(num_samples)

def lasso_kernel_shap_unpaired_2n(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = NewSHAP(n, game, interaction_strategy='LASSO kernel', paired_sampling=False, top_k=2*n)
    return estimator.shap_values(num_samples)
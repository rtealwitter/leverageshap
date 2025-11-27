from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.special import comb as binom

from .sampling import CoalitionSampler


class Game(ABC):
    def __init__(self, model, explicand):
        self.model = model
        self.explicand = explicand

    @abstractmethod
    def value(self, S):
        # S is a m by n binary matrix
        raise NotImplementedError

    @abstractmethod
    def edge_cases(self):
        raise NotImplementedError


class BaselineGame(Game):
    def __init__(self, model, baseline, explicand):
        super().__init__(model, explicand)
        self.baseline = baseline

    def value(self, S):
        # S is a m by n binary matrix
        inputs = self.baseline * (1 - S) + self.explicand * S
        return self.model.predict(inputs)

    def edge_cases(self):
        v0 = self.model.predict(self.baseline)
        v1 = self.model.predict(self.explicand)
        return v0, v1


class MarginalGame(Game):
    def __init__(self, model, background_data, explicand):
        super().__init__(model, explicand)
        self.background_data = background_data

    def value(self, S):
        N, *n = self.background_data.shape
        m = S.shape[0]

        data = np.expand_dims(self.background_data, axis=0)  # 1 x N x n
        S = np.expand_dims(S, axis=1)  # m x 1 x n
        inputs = data * (1 - S) + self.explicand * S
        inputs = inputs.reshape(-1, *n)  # (m * N) x n
        values = self.model.predict(inputs).reshape(m, N, -1)
        return np.mean(values, axis=1)

    def edge_cases(self):
        v0 = np.mean(self.model.predict(self.background_data), axis=0)
        v1 = self.model.predict(self.explicand)
        return v0, v1


class CustomMaskerGame(Game):
    """Game that uses a custom masker to compute the value of the game.

    Args:
        model: The model to evaluate.
        masker: A function that takes a m x n binary matrix and an explicand n-vector
            It returns a m x N x n matrix.
            Unline in the `shap` library, the masker needs to accept batched masks.
        explicand: The explicand to explain.
    """

    def __init__(self, model, masker, explicand):
        super().__init__(model, explicand)
        # masker takes a m x n mask and returns a N x m x n matrix
        self.masker = masker
        self.n = explicand.shape[-1]

    def value(self, S):
        # S is a m by n binary matrix
        inputs = self.masker(S, self.explicand)
        m, N, *n = inputs.shape
        print(m, N, n)
        inputs = inputs.reshape(-1, *n)
        values = self.model.predict(inputs).reshape(m, N, -1)
        return np.mean(values, axis=1)

    def edge_cases(self):
        S0 = np.zeros((1, self.n))
        S1 = np.ones((1, self.n))
        v0 = self.value(S0)[0]
        v1 = self.value(S1)[0]
        return v0, v1


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
            print("Number of samples too small, setting to 6")
            num_samples = 6

        sampling_weights = np.ones(self.n-1)

        sampler = CoalitionSampler(n_players=self.n, sampling_weights=sampling_weights, pairing_trick=self.paired_sampling, random_state=42)
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
        v0, v1 = v0.reshape(1, -1), v1.reshape(1, -1)
        values_adjusted = values - (v1 - v0) * coalition_sizes[:, np.newaxis]/ self.n
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
            print(f"{yellow_start}Warning:{yellow_end} Singular matrix in Leverage SHAP with num_samples={num_samples} and num_players={self.n}, adding ridge regularization with alpha={sqrt_alpha**2}.")

        AtA_inv_Atb = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

        return AtA_inv_Atb + (v1 - v0) / self.n

def leverage_shap(masker, explicand, model, num_samples):
    if isinstance(masker, Callable):
        game = CustomMaskerGame(model, masker, explicand)
    elif isinstance(masker, np.ndarray):
        if masker.ndim == 1 or masker.shape[0] == 1:
            baseline = masker
            game = BaselineGame(model, baseline, explicand)
        else:
            background_data = masker
            game = MarginalGame(model, background_data, explicand)
    n = explicand.shape[-1]
    estimator = LeverageSHAP(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)

import numpy as np
import math
from scipy.special import comb as binom
from typing import Sequence, Tuple, TypeVar

T = TypeVar("T")

# # # HELPER FUNCTIONS # # #

def symmetric_round_even(x):
    # Function written by ChatGPT
    '''
    tgt is the nearest even integer to sum(x)
    Rounds the entries of x to integers so that the sum is tgt, and x is symmetric
    '''
    x = np.asarray(x, float); n = x.size
    tgt = int(np.round(x.sum()/2)*2)           # nearest even ≤ sum
    out = np.floor(x).astype(int)
    rem = tgt - out.sum()
    frac = x - np.floor(x)

    pairs = [(i, n-1-i, frac[i]+frac[n-1-i]) for i in range(n//2)]
    pairs.sort(key=lambda t: t[2], reverse=True)
    for i, j, _ in pairs:
        if rem < 2: break
        out[i] += 1; out[j] += 1; rem -= 2
    if n % 2 == 1 and rem == 1:                # give lone +1 to the center
        out[n//2] += 1; rem -= 1
    return out

def ith_combination(pool: Sequence[T], size: int, index: int) -> Tuple[T, ...]:
    # Function written by ChatGPT
    """
    Return the `index`-th k-combination of `pool` (0-based), in lexicographic
    order w.r.t. `pool`'s current order. Single pass, no while-loop.
    """
    n = len(pool)
    k = size

    if not (0 <= k <= n):
        raise ValueError("size must be between 0 and len(pool)")
    total = math.comb(n, k)
    if not (0 <= index < total):
        raise IndexError(f"index must be in [0, {total-1}] for C({n},{k})")

    combo = []
    for i in range(n):
        if k == 0:
            break

        # If we must take all remaining items
        if n - i == k:
            combo.extend(pool[i:i+k])
            k = 0
            break

        # Combinations that start by taking pool[i]
        c = math.comb(n - i - 1, k - 1)

        if index < c:
            combo.append(pool[i])
            k -= 1
        else:
            index -= c

    return tuple(combo)

def combination_generator(gen, n, s, num_samples):
    # Function written by ChatGPT
    """
    Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
    1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with ith_combination.
    2. If the number of combinations is large (converting to an int DOES cause an overflow error), randomly sample num_samples combinations directly with replacement.
    """
    num_combos = math.comb(n, s)
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)
        for i in indices:
            yield ith_combination(range(n), s, i)
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)

class CoalitionSampler:
    '''
    Samples coalitions according to:
    1. Given a budget, compute sampling probabilities per coalition size via closed-form inversion of the expected sample count function
    2. Sample coalitions of each size according to these probabilities
    '''
    def __init__(
        self,
        n_players: int,
        sampling_weights: np.ndarray,
        *,
        pairing_trick: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.n_players = n_players

        if len(sampling_weights) == n_players + 1:
            sampling_weights = sampling_weights[1:-1]
            print(f'Warning: sampling_weights should be of length n_players-1, ignoring first and last entries.')
        elif len(sampling_weights) == n_players:
            sampling_weights = sampling_weights[1:]
            print(f'Warning: sampling_weights should be of length n_players-1, ignoring first entry.')
        elif len(sampling_weights) != n_players - 1:
            raise ValueError(f"sampling_weights should be of length n_players-1, but got length {len(sampling_weights)}.")
        self.distribution = sampling_weights / np.min(sampling_weights)
        # Insert 0 for empty coalition size and full coalition size
        self.distribution = np.concatenate(([0.0], self.distribution, [0.0]))

        # Ensure smallest weight is 1
        self.pairing_trick = pairing_trick
        self.rng = np.random.default_rng(seed=random_state)

    def sampling_probs(self, sizes):
        return np.minimum(
            self.constant * self.distribution[sizes] / binom(self.n_players, sizes), 1
        )

    def get_sampling_probs(self, budget: int):
        # Function written by ChatGPT
        """
        Compute sampling probabilities without iteration by inverting the
        piecewise-linear function:
            E(c) = sum_k min(c * weights[k], comb_counts[k])
        where comb_counts[k] = C(n_players, k) and weights[k] = distribution[k].
        For any budget in [0, 2**n_players], this solves for a scale c such that
        E(c) ~= budget (up to floating-point error) and returns sampling_probs(sizes).
        """
        n = self.n_players
        sizes = np.arange(1, n)

        # Per-size caps = number of coalitions of that size
        comb_counts = binom(n, sizes).astype(float)          # C(n, k)
        # Per-size weights from the distribution (>= 1 by construction)
        weights = self.distribution[sizes].astype(float)

        # Target expected total, clipped to feasible range [0, 2^n]
        target_total = float(np.clip(budget, 0, np.sum(comb_counts)))
        if target_total == 0.0:
            self.constant = 0.0
            return self.sampling_probs(sizes)

        # Breakpoints where a term saturates: c >= comb_counts[k] / weights[k]
        saturation_thresholds = comb_counts / weights
        order = np.argsort(saturation_thresholds)
        comb_counts_sorted = comb_counts[order]
        weights_sorted = weights[order]
        thresholds_sorted = saturation_thresholds[order]

        # For the segment before saturating index k:
        #   E(c) = sum_{j<k} comb_counts_sorted[j] + c * sum_{j>=k} weights_sorted[j]
        saturated_prefix = np.concatenate(([0.0], np.cumsum(comb_counts_sorted[:-1])))
        weights_prefix = np.concatenate(([0.0], np.cumsum(weights_sorted[:-1])))
        remaining_weight = np.sum(weights_sorted) - weights_prefix

        # Expected total at each breakpoint (just as k would start saturating)
        expected_at_threshold = saturated_prefix + thresholds_sorted * remaining_weight

        # Find the first segment where target_total fits
        segment_idx = np.searchsorted(expected_at_threshold, target_total, side="left")

        if segment_idx >= len(thresholds_sorted):
            # Past all segments: all terms saturate
            scale = float(thresholds_sorted[-1])
        else:
            denom = remaining_weight[segment_idx]
            # If denom == 0, slope is zero (nothing left to grow) -> stick to the threshold
            scale = thresholds_sorted[segment_idx] if denom == 0 else \
                    min((target_total - saturated_prefix[segment_idx]) / denom,
                        thresholds_sorted[segment_idx])

        self.constant = float(scale)

    def add_one_sample(self, indices):
        self.coalitions_matrix[self._coalition_idx, indices] = 1
        self.sampled_coalitions_dict[tuple(sorted(indices))] = 1
        self._coalition_idx += 1 

    def sample(self, budget: int):
        # Budget is an EVEN number between 2 and 2^n
        assert budget >= 2, "Budget must be at least 2"
        budget = min(budget, 2**self.n_players)
        budget += budget % 2

        # Get sampling probabilities
        self.get_sampling_probs(budget-2) # minus 2 for empty and full coalitions
        sizes = np.arange(1, self.n_players)
        samples_per_size = symmetric_round_even(
            self.sampling_probs(sizes) * binom(self.n_players, sizes)
        )
        sampling_probs = samples_per_size / binom(self.n_players, sizes)

        # Initialize storage
        self.coalitions_matrix = np.zeros((budget, self.n_players), dtype=bool)
        self._coalition_idx = 0
        self.sampled_coalitions_dict = {}

        # Sample empty and full coalitions
        self.add_one_sample([])
        self.add_one_sample(list(range(self.n_players)))

        for idx, size in enumerate(sizes):
            if idx >= self.n_players//2 and self.pairing_trick:
                break  # Stop early because of pairing
            if self.pairing_trick and size == self.n_players // 2 and self.n_players % 2 == 0:
                combo_gen = combination_generator(
                    self.rng, self.n_players - 1, size - 1, samples_per_size[idx] // 2
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices) + [self.n_players - 1])
                    self.add_one_sample(list(set(range(self.n_players-1)) - set(indices)))
            else:
                combo_gen = combination_generator(
                    self.rng, self.n_players, size, samples_per_size[idx]
                )
                for indices in combo_gen:
                    self.add_one_sample(list(indices))
                    if self.pairing_trick:
                        self.add_one_sample(
                            list(set(range(self.n_players)) - set(indices))
                        )

        coalition_sizes = np.sum(self.coalitions_matrix, axis=1)
        # Assign 1 to sizes of 0 and n
        self.sampling_probabilities = np.ones(self.coalitions_matrix.shape[0])
        filter_idx = (coalition_sizes > 0) & (coalition_sizes < self.n_players)
        self.sampling_probabilities[filter_idx] = sampling_probs[coalition_sizes[filter_idx]-1]
        self.sampling_adjustment_weights = np.ones(self.coalitions_matrix.shape[0])
        self.sampling_adjustment_weights[filter_idx] = 1 / sampling_probs[coalition_sizes[filter_idx]-1]
import itertools
import math
import numpy as np

from leverageshap.benchmark import build_gamma_labels


def test_synthetic_gamma_reference_matches_shapley_definition():
    # Evaluate the independent marginal-contribution definition on a tiny game.
    n = 4
    labels = build_gamma_labels(n, 0.5)
    value = lambda mask: labels['v'][labels['correspondence'][mask]]
    exact = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            for subset in itertools.combinations(others, size):
                mask = sum(1 << j for j in subset)
                exact[i] += (value(mask | (1 << i)) - value(mask)) / (n * math.comb(n - 1, size))
    np.testing.assert_allclose(labels['true_shap_values'], exact, atol=1e-12)
    assert np.isclose(np.sum(labels['true_shap_values']), 1)

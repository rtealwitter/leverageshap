import lightgbm as lgb
import numpy as np
from shapiq.approximator.sampling import CoalitionSampler
import math
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, cast, get_args
from collections.abc import Callable
from typing import Any


def proxyspex(game, budget, n, top_k: int) -> dict[tuple[int, ...], float]:
    random_state = 0

    # Sample budget uniform coalitions (boolean lists) from game
    uniform_sampler = CoalitionSampler(
        n_players=n,
        sampling_weights=np.array([math.comb(n, i) for i in range(n + 1)], dtype=float),
        pairing_trick=True,
        random_state=random_state,
    )
    uniform_sampler.sample(budget)

    train_X = pd.DataFrame(
        uniform_sampler.coalitions_matrix,
        columns=np.array([f"f{i}" for i in range(n)]),
    )
    train_y = game(uniform_sampler.coalitions_matrix)

    base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=random_state)

    param_grid = {
        "max_depth": [3, 5],
        "max_iter": [500, 1000],
        "learning_rate": [0.01, 0.1],
    }

    # Set up GridSearchCV with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        verbose=0,
        n_jobs=1,
    )

    grid_search.fit(train_X, train_y)

    best_model = grid_search.best_estimator_

    initial_transform = lgboost_to_fourier(best_model.booster_.dump_model())

    return top_k_odd_interactions(initial_transform, top_k)


def lgboost_to_fourier(model_dict: dict[str, Any]) -> dict[tuple[int, ...], float]:
    """Extracts the aggregated Fourier coefficients from an LGBoost model dictionary.

    This method iterates over all trees in the LightGBM ensemble, computes the
    Fourier coefficients for each individual tree using the `_lgboost_tree_to_fourier`
    helper method, and then sums these coefficients to get the final Fourier
    representation of the complete model.

    Args:
    model_dict: A dictionary representing the trained LGBoost model, as
        produced by `model.booster_.dump_model()`.

    Returns:
        A dictionary that maps interaction tuples (representing Fourier frequencies)
        to their aggregated Fourier coefficients.
    """
    aggregated_coeffs = defaultdict(float)

    for tree_info in model_dict["tree_info"]:
        tree_coeffs = lgboost_tree_to_fourier(tree_info)
        for interaction, value in tree_coeffs.items():
            aggregated_coeffs[interaction] += value

    # Convert defaultdict to a standard dict, removing zero-valued coefficients
    return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

def lgboost_tree_to_fourier(tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
    """Recursively strips the Fourier coefficients from a single LGBoost tree.

    This method traverses a tree's structure, as provided by LightGBM's `dump_model`
    method, and computes the Fourier representation of the piecewise-constant
    function that the tree defines. The logic is adapted from the work by Gorji et al. (2024).

    Args:
        tree_info: A dictionary representing a single decision tree from an LGBM model.

    Returns:
        A dictionary mapping interaction tuples to their corresponding coefficients for
        the single tree.

    References:
        Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
        "SHAP values via sparse Fourier representation"
        arXiv preprint arXiv:2410.06300 (2024).
    """

    def combine_coeffs(
        left_coeffs: dict[tuple[int, ...], float],
        right_coeffs: dict[tuple[int, ...], float],
        feature_idx: int,
    ) -> dict[tuple[int, ...], float]:
        """Combines Fourier coefficients from the left and right children of a split node."""
        combined_coeffs = {}
        all_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())

        for interaction in all_interactions:
            left_val = left_coeffs.get(interaction, 0.0)
            right_val = right_coeffs.get(interaction, 0.0)
            combined_coeffs[interaction] = (left_val + right_val) / 2

            new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
            combined_coeffs[new_interaction] = (left_val - right_val) / 2
        return combined_coeffs

    def dfs_traverse(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Performs a depth-first traversal of the tree to compute coefficients."""
        # Base case: if the node is a leaf, its function is a constant.
        if "leaf_value" in node:
            # The only non-zero coefficient is for the empty interaction (the bias term).
            return {(): node["leaf_value"]}
        # Recursive step: if the node is a split node.
        left_coeffs = dfs_traverse(node["left_child"])
        right_coeffs = dfs_traverse(node["right_child"])
        feature_idx = node["split_feature"]
        return combine_coeffs(left_coeffs, right_coeffs, feature_idx)

    return dfs_traverse(tree_info["tree_structure"])


def top_k_odd_interactions(four_dict: dict[tuple[int, ...], float], k: int) -> dict[tuple[int, ...], float]:
    """Return the top-k Fourier coefficients whose interaction keys have an odd
    cardinality greater than 1.

    Parameters
    ----------
    four_dict
        Mapping from interaction tuples to Fourier coefficients.
    k
        Number of top interactions to return.

    Returns
    -------
    dict
        Dictionary of the selected interaction tuples mapped to their
        coefficients, ordered by descending magnitude.
    """
    # Sort by absolute coefficient magnitude descending
    items = sorted(four_dict.items(), key=lambda iv: abs(iv[1]), reverse=True)

    selected: list[tuple[tuple[int, ...], float]] = []
    for key, val in items:
        if len(key) > 1 and (len(key) % 2 == 1):
            selected.append((key, val))
            if len(selected) >= k:
                break

    return {k: v for k, v in selected}


def refine(
        four_dict: dict[tuple[int, ...], float],
        train_X: np.ndarray,
        train_y: np.ndarray,
        se_threshold: float = 0.95,
    ) -> dict[tuple[int, ...], float]:
        """Refines the estimated Fourier coefficients using a Ridge regression model.

        This method takes an initial set of estimated Fourier coefficients and refines them to
        better fit the observed game values. It first identifies the most significant
        coefficients by keeping those that contribute to 95% of the total "energy" (sum of
        squared Fourier coefficients, excluding the baseline). Then, it constructs a new feature matrix
        based on the Fourier basis functions corresponding to these significant interactions.
        Finally, it fits a `RidgeCV` model to re-estimate the values of these coefficients,
        effectively fine-tuning them against the training data.

        Args:
            four_dict: A dictionary mapping interaction tuples to their initial estimated
                Fourier coefficient values.
            train_X: The training data matrix where rows are coalitions (binary vectors) and
                columns are players.
            train_y: The corresponding game values for each coalition in `train_X`.

        Returns:
            A dictionary containing the refined Fourier coefficients for the most significant
            interactions.
        """
        n = train_X.shape[1]
        four_items = list(four_dict.items())
        list_keys = [item[0] for item in four_items]
        four_coefs = np.array([item[1] for item in four_items])

        nfc_idx = list_keys.index(()) if () in list_keys else None

        four_coefs_for_energy = np.copy(four_coefs)
        if nfc_idx is not None:
            four_coefs_for_energy[nfc_idx] = 0
        four_coefs_sq = four_coefs_for_energy**2
        tot_energy = np.sum(four_coefs_sq)
        sorted_four_coefs_sq = np.sort(four_coefs_sq)[::-1]
        cumulative_energy_ratio = np.cumsum(sorted_four_coefs_sq / tot_energy)
        thresh_idx_95 = np.argmin(cumulative_energy_ratio < se_threshold) + 1
        thresh = np.sqrt(sorted_four_coefs_sq[thresh_idx_95])

        four_dict_trunc = {
            tuple(int(i in k) for i in range(n)): v for k, v in four_dict.items() if abs(v) > thresh
        }
        support = np.array(list(four_dict_trunc.keys()))

        X = np.real(np.exp(train_X @ (1j * np.pi * support.T)))
        reg = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False).fit(X, train_y)

        regression_coefs = dict(
            zip([tuple(s.astype(int)) for s in support], reg.coef_, strict=False)
        )
        return {tuple(i for i, x in enumerate(k) if x): v for k, v in regression_coefs.items()}



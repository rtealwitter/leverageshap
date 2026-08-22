import shap
import numpy as np
import scipy.special
import itertools
import pandas as pd

# Restored verbatim from commit 0de0a80 (paper-era code, predates the package
# simplification). Generates a synthetic set function on `num_features`
# players so gamma (Corollary 4.2) can be swept by hand via the `alpha`
# mixing parameter, independent of any real dataset's model. Used by the
# 'Synthetic' entry in `dataset_loaders` below; the actual gamma-sweep
# experiments in `benchmark_gamma` (leverageshap/benchmark.py) build labels
# directly with `build_gamma_labels` instead of going through this loader.
def synthetic(num_features=15):
    binary = np.zeros((2**num_features-2, num_features))
    idx = 0
    for s in range(1, num_features):
        for indices in itertools.combinations(range(num_features), s):
            binary[idx, list(indices)] = 1
            idx += 1
    num_ones = np.sum(binary, axis=1)
    inv_weights = num_ones * (num_features - num_ones) * scipy.special.binom(num_features, num_ones)
    weights = 1 / inv_weights
    Z = binary * weights[:, np.newaxis] # each row is w(||z||_1) z^T
    P = np.eye(num_features) - 1/num_features # projection matrix to remove all ones component
    A = Z @ P # each row is w(||z||_1) z^T P
    xstar = np.random.randn(num_features)
    ystar = A @ xstar
    weight_prob = weights / np.sum(weights)
    leverage = 1 / scipy.special.binom(num_features, num_ones)
    leverage_prob = leverage / np.sum(leverage)
    leverage_smaller = leverage_prob < weight_prob
    # Add noise
    noise = np.random.randn(2**num_features-2) * leverage_smaller
    # Convert to pandas dataframe
    X = pd.DataFrame(binary, columns=[f'Feature {i}' for i in range(num_features)])
    y = pd.Series(ystar + noise, name='Target')
    return X, y

dataset_loaders = {
    'Adult' : shap.datasets.adult,
    'California' : shap.datasets.california,
    'Communities' : shap.datasets.communitiesandcrime,
    'Correlated' : shap.datasets.corrgroups60,
    'Diabetes' : shap.datasets.diabetes,
    'Independent' : shap.datasets.independentlinear60,
    'IRIS' : shap.datasets.iris,
    'NHANES' : shap.datasets.nhanesi,
    'Synthetic' : synthetic,
}

def load_dataset(dataset_name):
    X, y = dataset_loaders[dataset_name]()
    # Remove nan values
    X = X.fillna(X.mean())
    return X, y

def load_input(X, seed=None, is_synthetic=False):
    # is_synthetic restored from commit 0de0a80: for the 'Synthetic' dataset
    # (see `synthetic()` above) the baseline/explicand are the all-zeros and
    # all-ones coalitions rather than a data-mean baseline and a sampled row,
    # matching how `synthetic()`'s labels are indexed by the binary_Z rows.
    # This is a new branch on top of the current, bug-fixed default path
    # below (float64 copies, see comment there); the default path itself is
    # unchanged for every existing caller (is_synthetic defaults to False).
    if is_synthetic:
        baseline = np.zeros((1, X.shape[1]))
        explicand = np.ones((1, X.shape[1]))
        return baseline, explicand
    if seed is not None:
        np.random.seed(seed)
    baseline = np.array(X.mean().values, dtype='float64').reshape(1, -1)
    explicand_idx = np.random.choice(X.shape[0])
    # Copy as float64: on mixed-dtype frames `.values` can be a read-only view
    # under recent pandas, which made the in-place edit below crash (NHANES).
    explicand = np.array(X.iloc[explicand_idx].values, dtype='float64').reshape(1, -1)
    for i in range(explicand.shape[1]):
        while baseline[0, i] == explicand[0, i]:
            explicand_idx = np.random.choice(X.shape[0])
            explicand[0,i] = X.iloc[explicand_idx, i]
    return baseline, explicand
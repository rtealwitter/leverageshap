import matplotlib.pyplot as plt
import scienceplots
import itertools
from .estimators import *
from .datasets import *
from .utils import fancy_round
import numpy as np
import xgboost as xgb
import os
from tqdm import tqdm
import scipy
import ast
import re

# Every line of output files contains a dictionary with the following keys
# 'sample_size': number of samples used to estimate SHAP values
# 'noise': standard deviation of noise added to the labels
# 'shap_error': mean squared error between estimated and true SHAP values
# 'weighted_error' (optional): ||Ax- b||^2 / ||Ax* - b||^2 where x* is the true SHAP values and x is the estimated SHAP values
# 'gamma' (optional, small n only): ||A x* - b||^2 / ||A x*||^2 where x* is the true SHAP values (Corollary 4.2 in the paper)

def build_full_linear_system(baseline, explicand, model):
    n = baseline.shape[1]
    binary_Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            binary_Z[idx, list(indices)] = 1
            idx += 1
    binary_Z1_norm = np.sum(binary_Z, axis=1)
    inv_sqrt_weights = np.sqrt(binary_Z1_norm * (n - binary_Z1_norm) * scipy.special.binom(n, binary_Z1_norm))

    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P
    inputs = baseline * (1 - binary_Z) + explicand * binary_Z
    v1 = model.predict(explicand)
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    y = (vz - v0) / inv_sqrt_weights
    b = y - Z.sum(axis=1) * (v1 - v0) / n
    return {'A': A, 'b': b}

def get_dataset_size(dataset):
    # 'Synthetic_<n>' is the dataset-name convention benchmark_gamma() (below)
    # writes its output/*.csv rows under -- restored from commit 0de0a80 so
    # load_results() can size those runs without loading an actual dataset.
    if 'Synthetic_' in dataset:
        return int(dataset.split('_')[1])
    X, y = load_dataset(dataset)
    return X.shape[1]

# numpy >= 2.0 prints scalars as np.float64(0.5); rows are persisted with str(dict) and read back with
# ast.literal_eval, so convert numpy scalars to native Python types before writing (and tolerate the
# wrapped form when reading files written before this fix).
_NUMPY_SCALAR_REPR = re.compile(r'np\.(?:float\d*|int\d*|bool_?)\(([^()]*)\)')

def to_native(row):
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}


def read_file(dataset, estimator, x_name, y_name, constraints={}):
    filename = f'output/{dataset}_{estimator}.csv'
    if not os.path.exists(filename): return {}
    results = {}
    with open(filename, 'r') as f:
        for line in f:
            line = _NUMPY_SCALAR_REPR.sub(r'\1', line.strip())
            if not line: continue
            try:
                dict = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                # torn final line from an interrupted job
                print(f'Skipping unparseable line in {filename}')
                continue
            add = True
            for key, value in constraints.items():
                if dict[key] != value:
                    add = False
            if add:
                try:
                    x, y = dict[x_name], dict[y_name]
                    if x not in results:
                        results[x] = []
                    results[x].append(y)
                except KeyError:
                    pass
    return results

def load_results(datasets, x_name, y_name, constraints, estimator_names=estimators.keys(), is_actual_sample_size=False):
    results_by_dataset = {}
    original_sample_size = constraints.get('sample_size', 1)
    for dataset in datasets:
        n = get_dataset_size(dataset)
        if 'sample_size' in constraints and not is_actual_sample_size:
            constraints['sample_size'] = int(original_sample_size * n)
        results_by_estimator = {}
        for estimator_name in estimator_names:
            if estimator_name == 'Tree SHAP':
                continue
            results = read_file(dataset, estimator_name, x_name, y_name, constraints)
            if results != {}:
                results_by_estimator[estimator_name] = results
        if results_by_estimator != {}:
            results_by_dataset[dataset] = results_by_estimator
    return results_by_dataset

def compute_weighted_error(baseline, explicand, model, shap_values):
    n = baseline.shape[1]
    Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            Z[idx, list(indices)] = 1
            idx += 1
    Z1_norm = np.sum(Z, axis=1)
    inv_weights = Z1_norm * (n - Z1_norm) * scipy.special.binom(n, Z1_norm)
    weights = 1 / inv_weights
    inputs = baseline * (1 - Z) + explicand * Z
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    return np.sum(weights * (shap_values @ Z.T - (vz - v0)) ** 2)

markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']

cbcolors = ['#88CCEE', '#332288', '#117733', '#CC6677', '#44AA99', '#AA4499', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466', '#4477AA']

def visualize_predictions(datasets, include_estimators, filename, seed=0):
    # seed: passed to load_input so the explicand (and numpy's global RNG, used by
    # the SHAP-library estimators) is the same every time the figure is regenerated.
    plt.clf()
    plt.style.use('science')
    # Three panels per row; as many rows as the estimator list needs (the main
    # figure has 3 estimators, the ablation figure 8).
    num_panels = len(include_estimators)
    row_num = (num_panels + 2) // 3
    fig, axs = plt.subplots(row_num, 3, figsize=(10, 3 * row_num), squeeze=False)
    for dataset_idx, dataset in enumerate(datasets):
        X, y = load_dataset(dataset)
        n = X.shape[1]
        num_samples = 5 * n
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
        model.fit(X, y)
        baseline, explicand = load_input(X, seed=seed)
        true_shap_values = estimators['Tree SHAP'](baseline, explicand, model, num_samples).flatten()
        # Ensure magnitude of true SHAP values is at most 1
        normalizing_scale = np.max(np.abs(true_shap_values))
        true_shap_values /= normalizing_scale
        i = 0
        for estimator_name, estimator in estimators.items():
            if estimator_name not in include_estimators:
                continue
            shap_values = estimator(baseline, explicand, model, num_samples).flatten()
            # Ensure magnitude of estimated SHAP values is at most 1
            shap_values /= normalizing_scale
            ax = axs[i // 3, i % 3]
            ax.scatter(true_shap_values, shap_values, alpha=0.5, marker=markers[dataset_idx], label=dataset + rf' ($n ={n}$)', color=cbcolors[dataset_idx])
            ax.set_title(estimator_name)
            i += 1
    
    # Hide the panels that no estimator uses (e.g. the ninth of a 3x3 grid).
    for j in range(num_panels, row_num * 3):
        axs[j // 3, j % 3].set_visible(False)

    for ax in axs.flatten()[:num_panels]:
        # Plot the line y = x
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], color='gray', alpha=0.5)

    # x label on the last used panel of every column, y label on the left column
    for col in range(3):
        used = [r for r in range(row_num) if r * 3 + col < num_panels]
        if used:
            axs[used[-1], col].set_xlabel(r'True Shapley Values ($\phi$)')
    for ax in axs[:, 0]:
        ax.set_ylabel(r'Predicted Shapley Values ($\tilde{\phi}$)')


    # Anchor the legend to the last used panel (a hidden panel would not draw it).
    axs.flatten()[num_panels - 1].legend(fancybox=True, bbox_to_anchor=(1,-.3), ncol=4)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.clf()

class NoisyModel:
    def __init__(self, model, noise_std):
        self.model = model
        self.noise_std = noise_std
        self.sample_count = 0

    def predict(self, X):
        self.sample_count += len(X)
        return self.model.predict(X) + np.random.normal(0, self.noise_std, X.shape[0])
    
    def get_sample_count(self):
        return self.sample_count

def run_small_setup(baseline, explicand, model, true_shap_values):
    linear_system = build_full_linear_system(baseline, explicand, model)
    best_weighted_error = np.sum((linear_system['A'] @ true_shap_values - linear_system['b'])**2)
    Aphi = linear_system['A'] @ true_shap_values
    gamma = np.sum((Aphi - linear_system['b'])**2) / np.sum((Aphi)**2)    
    normalized_gamma = gamma / np.sum((true_shap_values)**2)
    # Round to 2 significant figures
    normalized_gamma = float(f'{normalized_gamma:.2g}')
    return {'A': linear_system['A'], 'b': linear_system['b'], 'best_weighted_error': best_weighted_error, 'normalized_gamma': normalized_gamma, 'gamma': gamma}

def run_one_iteration(X, seed, dataset, model, sample_size, noise_std, num_runs, current_estimators):
    baseline, explicand = load_input(X, seed=seed)
    n = X.shape[1]
    is_small = 2**n <= 1e7
    # Compute the true SHAP values (assuming tree model)
    true_shap_values = estimators['Tree SHAP'](baseline, explicand, model, sample_size).flatten()

    small_setup = {}
     
    for estimator_name, estimator in current_estimators.items():        
        if estimator_name in ['Tree SHAP']:
            continue

        results = read_file(dataset, estimator_name, 'sample_size', 'shap_error', {'noise': noise_std, 'n': n})
        if results != {} and sample_size in results:
            if len(results[sample_size]) >= num_runs: continue
        noised_model = NoisyModel(model, noise_std)
        try:
            shap_values = estimator(baseline, explicand, noised_model, sample_size).flatten()
        except ValueError as e:
            print(f'SKIPPED (no row written): estimator {estimator_name}, dataset {dataset}, n={n}, sample_size={sample_size}, noise={noise_std}: {e}')
            continue

        filename = f'output/{dataset}_{estimator_name}.csv'
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('')

        with open(filename, 'a') as f:
            dict = {
                'sample_size': sample_size,
                'difference': noised_model.get_sample_count() - sample_size,
                'noise': noise_std,
                'n' : n,
            }
            shap_norm_sq = (true_shap_values**2).sum()
            dict['shap_error'] = ((shap_values - true_shap_values) ** 2).sum() / shap_norm_sq
            dict['shap_norm_sq'] = shap_norm_sq
            if is_small:
                if small_setup == {}:
                    small_setup = run_small_setup(baseline, explicand, model, true_shap_values)
                weighted_error = np.sum((small_setup['A'] @ shap_values - small_setup['b'])**2)
                dict['weighted_error'] = weighted_error / small_setup['best_weighted_error']
                dict['gamma'] = small_setup['gamma']
                dict['normalized_gamma'] = small_setup['normalized_gamma']
            f.write(str(to_native(dict)) + '\n')


def benchmark(num_runs, dataset, current_estimators, hyperparameter, hyperparameter_values, silent=False):              

    X, y = load_dataset(dataset)
    n = X.shape[1]
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    config = {'sample_size': 10*n, 'noise_std' : 0}
    for run_idx in tqdm(range(num_runs), disable=silent):
        for hyperparameter_value in hyperparameter_values:
            if hyperparameter == 'sample_size':
                hyperparameter_value = int(hyperparameter_value * n)
            config[hyperparameter] = hyperparameter_value
            run_one_iteration(X, run_idx * num_runs, dataset, model, sample_size=config['sample_size'], noise_std=config['noise_std'], num_runs=num_runs, current_estimators=current_estimators)

# ---------------------------------------------------------------------------
# Gamma (Corollary 4.2) experiments, restored verbatim from commit 0de0a80
# (before the package was simplified), with only two adaptations to match
# the current codebase: 'Official Tree SHAP' -> 'Tree SHAP' (the estimator
# dict key was renamed at HEAD) and the deliberate use of load_input's
# restored is_synthetic kwarg. compute_gamma, run_small_setup and
# run_one_iteration are otherwise untouched.
# ---------------------------------------------------------------------------

def compute_gamma(dataset, seed=42):
    # gamma = ||A phi* - b||^2 / ||A phi*||^2 for a real dataset's fitted
    # model (phi* = true Shapley values from Tree SHAP). Only defined for
    # datasets small enough to enumerate all 2^n-2 non-trivial coalitions.
    X, y = load_dataset(dataset)
    n = X.shape[1]
    is_small = 2**n <= 1e7
    if not is_small: return {}
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    baseline, explicand = load_input(X, seed=seed, is_synthetic=dataset=='Synthetic')
    true_shap_values = estimators['Tree SHAP'](baseline, explicand, model, num_samples=0).flatten()
    small_setup = run_small_setup(baseline, explicand, model, true_shap_values)
    return {
        'gamma': small_setup['gamma'],
        'normalized_gamma' : small_setup['normalized_gamma']
    }

class SyntheticModel:
    # A black-box model whose predictions are read directly out of a fixed
    # label vector `v`, indexed by the coalition's integer encoding through
    # `correspondence`. Used by benchmark_gamma so gamma can be dialed to a
    # target value via build_gamma_labels without fitting anything.
    def __init__(self, v, correspondence):
        self.v = v
        self.num_samples = 0
        self.correspondence = correspondence

    def predict(self, X):
        # X is a binary matrix
        # Get the integer represented in each row
        indices = np.sum(2**np.arange(X.shape[1]) * X, axis=1).astype(int)
        # Get the corresponding index in v
        self.num_samples += len(X)
        return self.v[[self.correspondence[i] for i in indices]]

    def get_sample_count(self):
        return self.num_samples

def build_gamma_labels(n, alpha):
    # Constructs a synthetic set function on n players whose gamma
    # (Corollary 4.2) is controlled by alpha: b is (1-alpha) times a unit
    # vector in the column span of A (gamma -> 0 as alpha -> 0, the
    # regression is consistent) plus alpha times a unit vector orthogonal to
    # every column of A (gamma -> infinity as alpha -> 1, b is unreachable).
    # Construct A
    binary_Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            binary_Z[idx, list(indices)] = 1
            idx += 1
    # Convert all rows to their integer form
    X = binary_Z
    indices = np.sum(2**np.arange(X.shape[1]) * X, axis=1).astype(int)
    correspondence = {0:0, 2**n-1:-1}
    for i in range(2**n-2):
        correspondence[indices[i]] = i + 1

    binary_Z1_norm = np.sum(binary_Z, axis=1)
    inv_sqrt_weights = np.sqrt(binary_Z1_norm * (n - binary_Z1_norm) * scipy.special.binom(n, binary_Z1_norm))
    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P

    # Perform QR decomposition of A
    Q, R = np.linalg.qr(A)

    # The last column of Q is orthogonal to all the columns of A
    # if A has full rank and is not square
    col_not_in_span = Q[:, -1]
    col_not_in_span = col_not_in_span / np.linalg.norm(col_not_in_span)

    # Check that r is orthogonal to the columns of A
    assert np.allclose(A.T @ col_not_in_span, 0)

    # Construct b as (1-alpha) * a column in span of A + alpha * a column not in span of A
    col_in_span = A[:, 0]
    col_in_span = col_in_span / np.linalg.norm(col_in_span)
    b = (1 - alpha) * col_in_span + alpha * col_not_in_span

    # Convert from b to y
    v1 = 1
    v0 = 0
    y = b + Z.sum(axis=1) * (v1 - v0) /n

    v = np.zeros(2**n)
    v[1:-1] = y * inv_sqrt_weights
    v[0] = v0
    v[-1] = v1

    # True SHAP values.
    # Solve Ax = b
    true_shap_values = np.linalg.lstsq(A, b, rcond=None)[0]
    best_weighted_error = np.sum((A @ true_shap_values - b)**2)

    gamma = np.sum((A @ true_shap_values - b)**2) / np.sum((A @ true_shap_values)**2)

    return {'v': v, 'true_shap_values': true_shap_values, 'best_weighted_error': best_weighted_error, 'correspondence': correspondence}

def benchmark_gamma(num_runs, n, include_estimators, sample_size, silent=False):
    baseline = np.zeros((1, n))
    explicand = np.ones((1, n))
    for run_idx in tqdm(range(num_runs), disable=silent):
        for alpha in [.2, .3, .4, .5, .6, .7, .8]:
            # Deterministic per (run_idx, alpha): seed depends only on the
            # run index and alpha (scaled to an integer via *100, since alpha
            # takes 2-decimal values), so every run_idx/alpha pair gets its
            # own reproducible draw of build_gamma_labels and every estimator
            # sees the same labels within that pair.
            seed = run_idx * num_runs + int(alpha * 100)
            np.random.seed(seed)
            gamma_labels = build_gamma_labels(n, alpha)

            is_small = 2**n <= 1e7

            small_setup = {}

            dataset = 'Synthetic_' + str(n)

            for estimator_name, estimator in estimators.items():
                if estimator_name not in include_estimators:
                    continue
                model = SyntheticModel(gamma_labels['v'], gamma_labels['correspondence'])
                results = read_file(dataset, estimator_name, 'alpha', 'shap_error', {'n': n})
                if results != {} and alpha in results:
                    if len(results[alpha]) >= num_runs: continue
                shap_values = estimator(baseline, explicand, model, sample_size).flatten()

                filename = f'output/{dataset}_{estimator_name}.csv'
                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        f.write('')

                with open(filename, 'a') as f:
                    dict = {
                        'sample_size': sample_size,
                        'difference': model.get_sample_count() - sample_size,
                        'noise': 0,
                        'n' : n,
                        'alpha' : alpha,
                    }
                    shap_norm_sq = (gamma_labels['true_shap_values'] ** 2).sum()
                    # float(...) casts below: under numpy>=2.0, repr(np.float64(x))
                    # is 'np.float64(x)' rather than plain 'x', which breaks
                    # read_file()'s ast.literal_eval round-trip on this line
                    # (confirmed via a smoke test: every row silently failed to
                    # parse back, i.e. every fresh run's data was unreadable).
                    # Casting to a native Python float only changes how the
                    # value is *serialized* to disk (numpy.float64 and Python
                    # float are both IEEE-754 doubles, so this loses no
                    # precision and changes no computed value) -- it is not
                    # restored verbatim from 0de0a80, since this incompatibility
                    # didn't exist under the pre-numpy-2.0 environment that code
                    # originally ran under. The identical pattern also exists in
                    # run_one_iteration() above, which per this restoration's
                    # scope is left untouched -- see the final report.
                    dict['shap_error'] = float(((shap_values - gamma_labels['true_shap_values']) ** 2).sum() / shap_norm_sq)
                    dict['shap_norm_sq'] = float(shap_norm_sq)
                    if is_small:
                        if small_setup == {}:
                            small_setup = run_small_setup(baseline, explicand, model, gamma_labels['true_shap_values'])
                        weighted_error = np.sum((small_setup['A'] @ shap_values - small_setup['b'])**2)
                        dict['weighted_error'] = float(weighted_error / gamma_labels['best_weighted_error'])
                        dict['gamma'] = float(small_setup['gamma'])
                    f.write(str(dict) + '\n')

def gamma_distribution_table(datasets, filename, num_runs=100, gammas=None):
    # NEW (not in 0de0a80): reproduces overleaf_paper/tables/gamma_distribution.tex
    # -- one row per real dataset with the 1st/2nd/3rd quartile of gamma
    # (Corollary 4.2) computed by compute_gamma() over num_runs seeds
    # (seed=0..num_runs-1, deterministic and reproducible). This is
    # distinct from benchmark_gamma() above, which sweeps a *synthetic*
    # set function's gamma via the alpha parameter; this table instead
    # asks how large gamma actually is for real fitted models.
    # gammas: optional {dataset: [gamma per seed]} computed elsewhere (e.g. by a
    # batch job); when given, compute_gamma is not called here.
    rows = []
    for dataset in datasets:
        values = gammas[dataset] if gammas is not None else [compute_gamma(dataset, seed=seed)['gamma'] for seed in range(num_runs)]
        n = get_dataset_size(dataset)
        lower = fancy_round(np.percentile(values, 25))
        median = fancy_round(np.percentile(values, 50))
        upper = fancy_round(np.percentile(values, 75))
        rows.append((dataset, n, lower, median, upper))
    with open(filename, 'w') as f:
        f.write('\\begin{tabular}{lcccc}\n')
        f.write('    \\toprule\n')
        f.write('    Dataset & $n$ & 1st Quartile & 2nd Quartile & 3rd Quartile \\\\\n')
        f.write('    \\midrule\n')
        for dataset, n, lower, median, upper in rows:
            f.write(f'    {dataset} & {n} & {lower} & {median} & {upper} \\\\\n')
        f.write('    \\bottomrule\n')
        f.write('\\end{tabular}\n')

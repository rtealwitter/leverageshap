import leverageshap as ls
import numpy as np
import sys

# Restored/rewritten to match the paper's structure (commit 0de0a80, before
# the package was simplified), using the current, bug-fixed estimator
# implementations. See leverageshap/estimators/ablations.py and README.md
# ("Benchmarking") for what each estimator name means.

small_n = ['IRIS', 'California', 'Diabetes', 'Adult']

big_n = ['Correlated', 'Independent', 'NHANES', 'Communities']

datasets = small_n + big_n

def get_hyperparameter_values(name):
    if name == 'noise_std':
        return [.5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
    elif name == 'sample_size':
        # Multipliers of n used throughout the paper, including as the
        # column headers of Table tab:error2kshap ($5n,10n,\dots,160n$).
        return [5, 10, 20, 40, 80, 160]
    else:
        raise ValueError(f'Unknown hyperparameter {name}')

# The 3 estimators shown in the paper's main body figures/tables.
main_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP']

# The paper's paired/leverage/Bernoulli ablation grid (a superset of
# main_estimators), shown in the appendix.
ablation_estimators = ['Kernel SHAP', 'Kernel SHAP Paired', 'Optimized Kernel SHAP', 'Permutation SHAP', 'Leverage SHAP (Unpaired)', 'Leverage SHAP wo Bernoulli', 'Leverage SHAP (Binomial)', 'Leverage SHAP']

# Usage:
#   python run.py                 -- re-plot/re-tabulate images/ and tables/
#                                     from whatever is already in output/*.csv
#   python run.py --benchmark     -- also regenerate output/*.csv (100 runs x
#                                     each dataset x sample_size/noise_std,
#                                     for every ablation_estimators entry --
#                                     slow, hours of compute)
#   python run.py --gamma         -- also print gamma (Corollary 4.2)
#                                     percentiles for the small_n real
#                                     datasets and write
#                                     tables/gamma_distribution.tex
#   python run.py --detailed      -- also regenerate images/main_detailed.pdf
#                                     and images/ablation_detailed.pdf
#   python run.py --probs         -- also regenerate images/sampling_prob.pdf
# Flags can be combined, e.g. `python run.py --benchmark --gamma`.

if '--benchmark' in sys.argv:
    ablation_estimators_dict = {
        name: ls.estimators[name] for name in ablation_estimators
    }
    num_runs = 100
    for dataset in datasets:
        print(dataset)
        for hyperparameter in ['sample_size', 'noise_std']:
            print(hyperparameter)
            ls.benchmark(num_runs, dataset, ablation_estimators_dict, hyperparameter, get_hyperparameter_values(hyperparameter), silent=False)

if '--gamma' in sys.argv:
    # Diagnostic: how large is gamma (Corollary 4.2) for real fitted models?
    # Also feeds tables/gamma_distribution.tex (reproduces
    # overleaf_paper/tables/gamma_distribution.tex).
    gammas = {x: [] for x in small_n}
    for seed in range(100):
        for dataset in small_n:
            gamma_run = ls.compute_gamma(dataset, seed=seed)
            gammas[dataset].append(gamma_run['gamma'])

        if seed % 10 != 0: continue
        print()
        for dataset in small_n:
            # dataset name, 1st quartile, median, 3rd quartile
            print(dataset, np.percentile(gammas[dataset], 25), np.median(gammas[dataset]), np.percentile(gammas[dataset], 75))

    ls.gamma_distribution_table(small_n, 'tables/gamma_distribution.tex', num_runs=100)

if '--probs' in sys.argv:
    ls.plot_probs([10, 100, 1000], folder='images/')

if '--detailed' in sys.argv:
    ls.visualize_predictions(datasets, main_estimators, filename='images/main_detailed.pdf')
    ls.visualize_predictions(datasets, ablation_estimators, filename='images/ablation_detailed.pdf')

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by number of samples
    x_name = 'sample_size'
    constraints = {'noise': 0}
    results = ls.load_results(datasets, x_name, y_name, constraints)
    for ending in ['png', 'pdf']:
        ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.{ending}', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)
        ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.{ending}', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators, plot_mean=True)

    # Performance by noise level
    x_name = 'noise'
    constraints = {'sample_size': 10}
    results = ls.load_results(datasets, x_name, y_name, constraints)
    for ending in ['png', 'pdf']:
        ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.{ending}', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)
        ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.{ending}', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators, plot_mean=True)

# Tables
for y_name in ['shap_error', 'weighted_error']:
    results = ls.load_results(datasets, 'sample_size', y_name, {'noise': 0, 'sample_size': 10})
    results_main = {}
    for dataset in results:
        # Guard against a dataset missing one of main_estimators (e.g. a
        # partial/incomplete --benchmark run): skip it rather than KeyError
        # and abort the whole table -- the paper-era code (and the pre-
        # restore run.py) assumed complete data and would crash here.
        results_main[dataset] = {estimator: results[dataset][estimator] for estimator in main_estimators if estimator in results[dataset]}
    ls.one_big_table(results_main, f'tables/main_{y_name}.tex', error_type=y_name)
    ls.one_big_table(results, f'tables/ablation_{y_name}.tex', error_type=y_name)

    for dataset in results:
        ls.benchmark_table(results[dataset], f'tables/{dataset}-{y_name}.tex', print_md=False)

# Error ratio table (Table tab:error2kshap): needs every sample_size
# multiplier at once, so load_results is unconstrained on sample_size here
# (unlike the single-sample_size=10n Tables section above).
results_ratio = ls.load_results(datasets, 'sample_size', 'shap_error', {'noise': 0}, estimator_names=['Leverage SHAP', 'Optimized Kernel SHAP'])
ls.error_ratio_table(results_ratio, get_hyperparameter_values('sample_size'), 'tables/error_ratio.tex')

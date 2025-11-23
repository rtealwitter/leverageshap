import leverageshap as ls
import numpy as np

small_n = ['IRIS', 'California', 'Diabetes', 'Adult']

big_n = ['Correlated', 'Independent', 'NHANES', 'Communities']

def get_hyperparameter_values(name):
    #if name == 'noise_std': return [0]
    if name == 'noise_std':
        return [.5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
    elif name == 'sample_size':
        return [.25, .5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    else:
        raise ValueError(f'Unknown hyperparameter {name}')

# Collect all estimators exported by the package except 'Tree SHAP' (used as ground truth)
main_estimators = ['Leverage SHAP', 'Fourier SHAP', 'Uniform ProxySPEX paired', 'Uniform ProxySPEX unpaired', 'Kernel ProxySPEX paired', 'Kernel ProxySPEX unpaired']
datasets = small_n + big_n

if True:
    main_estimators = {
        name: ls.estimators[name] for name in main_estimators
    }
    num_runs = 10
    for dataset in small_n + big_n:
        print(dataset)
        for hyperparameter in ['sample_size']:#, 'noise_std']:
            print(hyperparameter)
            ls.benchmark(num_runs, dataset, main_estimators, hyperparameter, get_hyperparameter_values(hyperparameter), silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by number of samples
    x_name = 'sample_size'
    constraints = {'noise': 0}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    for ending in ['png', 'pdf']:
        ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}-SPEX.{ending}', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)

    # Performance by noise level
    x_name = 'noise'
    constraints = {'sample_size': 10}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    for ending in ['png', 'pdf']:
        ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}-SPEX.{ending}', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)

# Tables
for y_name in ['shap_error', 'weighted_error']:
    results = ls.load_results(small_n + big_n, 'sample_size', y_name, {'noise': 0, 'sample_size' : 8})
    results_main = {}
    for dataset in results:
        results_main[dataset] = {estimator : results[dataset][estimator] for estimator in main_estimators}
    ls.one_big_table(results_main, f'tables/main_{y_name}.tex', error_type=y_name)
        
    for dataset in results:
        ls.benchmark_table(results[dataset], f'tables/{dataset}-{y_name}.tex', print_md=False)
import leverageshap as ls
import numpy as np

small_n = ['IRIS', 'California', 'Diabetes', 'Adult']

big_n = ['Correlated', 'Independent', 'NHANES', 'Communities']

def get_hyperparameter_values(name):
    if name == 'noise_std': return [0]
    if name == 'noise_std':
        return [0, .5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
    elif name == 'sample_size':
        return [5, 10, 20, 40, 80, 160]
    else:
        raise ValueError(f'Unknown hyperparameter {name}')

# Debugging access to official SHAP estimators
#import logging
#log = logging.getLogger('shap')
#logging.basicConfig(level=logging.DEBUG)

ablation_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP', 'Kernel SHAP Paired', 'Leverage SHAP wo Bernoulli', 'Leverage SHAP wo Bernoulli, Paired', 'Matrix SHAP', 'Matrix SHAP wo Bernoulli', 'Matrix SHAP wo Bernoulli, Paired', 'Permutation SHAP']

main_estimators = ['Optimized Kernel SHAP', 'Leverage SHAP', 'Matrix SHAP']

datasets = small_n + big_n

if False:
    gammas = {x : [] for x in small_n}
    for seed in range(100):
        for dataset in small_n:
            gamma_run = ls.compute_gamma(dataset, seed=seed)
            gammas[dataset].append(gamma_run['gamma'])

        if seed % 10 != 0: continue
        print()
        for dataset in small_n:
            # print dataset name, 1st quartile, median, 3rd quartile
            print(dataset, np.percentile(gammas[dataset], 25), np.median(gammas[dataset]), np.percentile(gammas[dataset], 75))
            

ls.plot_probs([10,100,1000], folder='images/')

if False:

    ls.visualize_predictions(datasets, main_estimators, filename='images/main_detailed.pdf')
    #ls.visualize_predictions(datasets, ablation_estimators, filename='images/ablation_detailed.pdf')

if True:
    estimators = {'Monte Carlo' : ls.estimators['Monte Carlo']}
    num_runs = 90
    for dataset in small_n + big_n:
        print(dataset)
        for hyperparameter in ['sample_size', 'noise_std']:
            print(hyperparameter)
            ls.benchmark(num_runs, dataset, estimators, hyperparameter, get_hyperparameter_values(hyperparameter), silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by number of samples
    x_name = 'sample_size'
    constraints = {'noise': 0}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators, plot_mean=True)

    # Performance by noise level
    x_name = 'noise'
    constraints = {'sample_size': 10}
    results = ls.load_results(small_n + big_n, x_name, y_name, constraints)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators, plot_mean=True)

# Tables
for y_name in ['shap_error', 'weighted_error']:
    results = ls.load_results(small_n + big_n, 'sample_size', y_name, {'noise': 0, 'sample_size' : 10})
    ls.one_big_table(results, f'tables/ablation_{y_name}.tex', error_type=y_name)
    results_main = {}
    for dataset in results:
        results_main[dataset] = {estimator : results[dataset][estimator] for estimator in main_estimators}
    ls.one_big_table(results_main, f'tables/main_{y_name}.tex', error_type=y_name)
        
    #for dataset in results:
    #    ls.benchmark_table(results[dataset], f'tables/{dataset}-{y_name}.tex', print_md=False)
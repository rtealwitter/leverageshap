import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from .estimators import estimators
from .benchmark import get_dataset_size

linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5,(10,3)), (0,(1,1)), (0,(5,10)),(0,(5,1)), (0,(3,10,1,10)), (0,(3,5,1,5)), (0,(3,1,1,1)), (0,(3,5,1,5,1,5)), (0,(3,10,1,10,1,10)), (0,(3,1,1,1,1,1))]

cbcolors = ['#332288', '#88CCEE', '#CC6677', '#117733', '#44AA99', '#AA4499', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466', '#4477AA']

linestyles_lookup = {name: linestyles[i % len(linestyles)] for i, name in enumerate(estimators.keys())}

cbcolors_lookup = {name: cbcolors[i % len(cbcolors)] for i, name in enumerate(estimators.keys())}

name_lookup = {
    'shap_error': r'Error in $\ell_2$-norm', #r'Shapley Error $\left(\| \tilde{{\phi}} - {\phi}\|_2^2\right)$',
    'weighted_error': r'Error in Linear Objective',# $\left( \frac{\| {A} \tilde{{\phi}} - {b} \|_2^2}{\| {A} {\phi} - {b} \|_2^2} \right)$',
    'sample_size': r'Sample Size $(m)$',
    'noise': r'Standard Deviation of Noise $(\sigma)$',
    'gamma': r'$\gamma$',# $\left( \frac{\| {b} \|_2^2}{\| {A} {\phi} \|_2^2} \right)$',
}

def plot_with_subplots(results, x_name, y_name, filename=None, log_x=True, log_y=True, plot_mean=False, include_estimators=estimators):
    plt.clf()
    num_datasets = len(results)
    num_rows = 1 if num_datasets <= 4 else 2
    dims = (num_rows, num_datasets // num_rows + num_datasets % num_rows)
    fig, axs = plt.subplots(*dims, figsize=(num_datasets /num_rows * 4, 3 * num_rows))
    for i, (dataset, results_by_dataset) in enumerate(results.items()):
        n = get_dataset_size(dataset)
        if num_datasets > 4:
            ax = axs[i // 4, i % 4]
        else:
            ax = axs[i] if num_datasets > 1 else axs

        for estimator_name, results_by_estimator in results_by_dataset.items():
            if estimator_name not in include_estimators: continue
            x_values = list(results_by_estimator.keys())
            x_values = sorted(x_values)
            y_mean = [np.mean(results_by_estimator[x]) for x in x_values]
            y_median = np.array([np.median(results_by_estimator[x]) for x in x_values])
            y_upper = np.array([np.percentile(results_by_estimator[x], 75) for x in x_values])
            y_lower = np.array([np.percentile(results_by_estimator[x], 25) for x in x_values])
            if plot_mean:
                ax.plot(x_values, y_mean, label=estimator_name, linestyle=linestyles_lookup[estimator_name], color=cbcolors_lookup[estimator_name])
            else:
                ax.plot(x_values, y_median, label=estimator_name, linestyle=linestyles_lookup[estimator_name], color=cbcolors_lookup[estimator_name])
                ax.fill_between(x_values, y_lower, y_upper, alpha=0.2, color=cbcolors_lookup[estimator_name])

        if '_' in dataset:
            dataset, _ = dataset.split('_')
        ax.set_title(rf'{dataset} $(n = {n})$')
        if log_x: ax.set_xscale('log')
        if log_y: ax.set_yscale('log') 
        if 2**n < ax.get_xlim()[1] and 2**n > ax.get_xlim()[0]:
            ax.axvline(x=2**n, color='r', linestyle='solid')
            ax.annotate(r'$2^n$', xy=(2**n, ax.get_ylim()[1]), xytext=(2,-10), textcoords='offset points', color='r')

        if num_rows == 1 or i >= 4:
            ax.set_xlabel(name_lookup[x_name])
        if i % 4 == 0:
            ax.set_ylabel(name_lookup[y_name])

    num_labels = len(plt.legend().get_texts())
    plt.legend(fancybox=True, bbox_to_anchor=(1,-.3), ncol=num_labels+1)
    if filename is not None:
        plt.savefig(filename, dpi=1000, bbox_inches='tight')
    else:
        plt.show()

def plot_data(results, dataset, filename=None, exclude=[], weighted_error=False):
    plt.clf()
    num = 0
    for name, data in results.items():
        if name in exclude or name == 'n':
            continue
        sample_sizes = list(data.keys())
        mean = np.array([np.mean(data[sample_size]) for sample_size in sample_sizes])
#        std = np.array([np.std(data[sample_size]) for sample_size in sample_sizes])
        plt.plot(sample_sizes, mean, label=name, linestyle=linestyles_lookup[name], color=cbcolors_lookup[name])
#        plt.fill_between(sample_sizes, mean - std, mean + std, alpha=0.2)
        num +=1
    # Put legend outside of plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'{dataset} Benchmark n = {results["n"]}')
    plt.xscale('log')   
    if not weighted_error:
        plt.ylabel('Shapley 2-norm Error')
        plt.yscale('log')
    else:
        plt.ylabel('Weighted Error')
    plt.xlabel('Sample Size')
    # Plot vertical red line at 2^n if 2^n is less than the plt.xlim
    if 2**results['n'] < plt.xlim()[1]:
        plt.axvline(x=2**results['n'], color='r', linestyle='--')
    if filename is not None:
        plt.savefig(filename, dpi=1000, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_probs(ns, folder=None):
    plt.clf()
    # Set figsize
    # Three subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    ns = [10, 100, 1000]
    for idx, n in enumerate(ns):
        s = np.arange(1, n)
        kernel_weight = 1 / (s * (n - s))
        kernel_prob = kernel_weight / np.sum(kernel_weight)
        leverage_weight = np.ones_like(s)
        leverage_prob = leverage_weight / np.sum(leverage_weight)
        axs[idx].plot(s, kernel_prob, label='Kernel SHAP', color='b')
        axs[idx].plot(s, leverage_prob, label='Leverage SHAP', linestyle='--', color='g')
        axs[idx].set_title(f'n = {n}')
        axs[idx].set_yscale('log')
        axs[idx].set_xlabel('Subset Size')
        if idx == 0:
            axs[idx].set_ylabel('Probability')
    plt.legend(loc='center left', bbox_to_anchor=(0, -.3), ncol=2)
    plt.suptitle('Kernel SHAP and Leverage SHAP Probability Distributions', y=1.05, fontsize=16)
    filename = f'{folder}sampling_prob.pdf'
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()

def plot_sampled_sizes(n, m, folder=None):
    s = np.arange(1, n)
    kernel_weight = 1 / (s * (n - s))
    kernel_prob = kernel_weight / np.sum(kernel_weight)
    kernel_sampled = np.random.choice(s, m, p=kernel_prob)
    leverage_weight = np.ones_like(s)
    leverage_prob = leverage_weight / np.sum(leverage_weight)
    leverage_sampled = np.random.choice(s, m, p=leverage_prob)

    plt.hist(kernel_sampled, alpha=0.5, label='KernelSHAP', color='b')
    plt.hist(leverage_sampled, alpha=0.5, label='LeverageSHAP', color='g')
    plt.legend()
    plt.title('Sampled Subset Sizes')
    plt.xlabel('Subset Size')
    plt.ylabel('Frequency')
    filename = f'{folder}sampled_sizes_{n}_{m}.pdf'
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.close()

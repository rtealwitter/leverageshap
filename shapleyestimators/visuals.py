import matplotlib.pyplot as plt
import numpy as np

def plot_data(results, dataset, filename=None, exclude=[], weighted_error=False):
    plt.clf()
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (0,(1,10)), (0,(1,1)), (0,(5,10)),(0,(5,1)), (0,(3,10,1,10)), (0,(3,5,1,5)), (0,(3,1,1,1)), (0,(3,5,1,5,1,5)), (0,(3,10,1,10,1,10)), (0,(3,1,1,1,1,1))]
    num = 0
    for name, data in results.items():
        if name in exclude or name == 'n':
            continue
        sample_sizes = list(data.keys())
        mean_errors = [np.mean(data[sample_size]) for sample_size in sample_sizes]
        plt.plot(sample_sizes, mean_errors, label=name, linestyle=linestyles[num % len(linestyles)])
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

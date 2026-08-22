import leverageshap as ls
import sys

# Restored from commit 0de0a80 (before the package was simplified), using
# the current, bug-fixed estimator implementations. Sweeps gamma
# (Corollary 4.2) on synthetic set functions of size n via build_gamma_labels
# (see leverageshap/benchmark.py), independent of any real dataset.

# The 0de0a80 original listed 'Leverage SHAP wo Bernoulli, Paired' here, a
# (paired_sampling=False, leverage_sampling=True, bernoulli_sampling=False)
# ablation config that was never actually wired into that commit's own
# estimators dict (it was commented out there too) and predates the current,
# bug-fixed unpaired variant. Per the same decision documented in
# leverageshap/estimators/ablations.py and leverage_shap.py, we do not
# resurrect that dead with-replacement code path here; 'Leverage SHAP
# (Unpaired)' (wrapping the current deterministic sampler with
# paired_sampling=False) stands in as the unpaired comparison point instead.
ablation_estimators = ['Kernel SHAP', 'Kernel SHAP Paired', 'Optimized Kernel SHAP', 'Leverage SHAP (Unpaired)', 'Leverage SHAP wo Bernoulli', 'Leverage SHAP (Binomial)', 'Leverage SHAP']

main_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP']

ns = [10, 12, 14, 16]
sample_size = 500
num_runs = 100

# Usage:
#   python run-gamma.py              -- re-plot images/*gamma*.pdf from
#                                        whatever is already in output/*.csv
#   python run-gamma.py --benchmark  -- also regenerate the underlying
#                                        output/Synthetic_<n>_*.csv data
#                                        (100 runs x 7 alphas x n in ns x
#                                        each ablation_estimators entry)
if '--benchmark' in sys.argv:
    for n in ns:
        ls.benchmark_gamma(num_runs, n, ablation_estimators, sample_size=sample_size, silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by gamma
    x_name = 'gamma'
    constraints = {'sample_size': sample_size, 'noise': 0}
    results = ls.load_results([f'Synthetic_{n}' for n in ns], x_name, y_name, constraints, is_actual_sample_size=True)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/main_gamma_{y_name}.pdf', log_x=True, log_y=True, include_estimators=main_estimators, plot_mean=False)
    ls.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_gamma_{y_name}.pdf', log_x=True, log_y=False, include_estimators=ablation_estimators, plot_mean=True)

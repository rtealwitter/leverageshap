from .official_shap import *
from .leverage_shap import *
from .ablations import *

# Keyed by the paper's names. 'Kernel SHAP', 'Kernel SHAP Paired',
# 'Leverage SHAP wo Bernoulli', and 'Leverage SHAP (Binomial)' are the
# paper-era ablation estimators restored (see ablations.py) from commit
# 0de0a80 for the ablation/gamma experiments in the paper's appendix.
# 'Optimized Kernel SHAP' is the SHAP-library estimator (was keyed as
# 'Kernel SHAP' pre-restore -- renamed so that name is free for the paper's
# actual vanilla "Kernel SHAP" ablation estimator above).
estimators = {
    'Kernel SHAP': kernel_shap,
    'Kernel SHAP Paired': kernel_shap_paired,
    'Optimized Kernel SHAP': official_kernel_shap,
    'Permutation SHAP': official_permutation_shap,
    'Leverage SHAP': leverage_shap,
    'Leverage SHAP (Unpaired)': leverage_shap_unpaired,
    'Leverage SHAP wo Bernoulli': leverage_shap_wo_bernoulli,
    'Leverage SHAP (Binomial)': leverage_shap_binomial,
    'Tree SHAP': official_tree_shap,
}

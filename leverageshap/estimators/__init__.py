from .official_shap import *
from .leverage_shap import *
from .regression_msr import *
from .new import *

estimators = {
    'Kernel SHAP': official_kernel_shap,
    'Permutation SHAP': official_permutation_shap,
    'Leverage SHAP': leverage_shap,
    'Tree SHAP': official_tree_shap,
    'Regression MSR': regression_msr,
    # Fourier SHAP
    'Fourier SHAP': fourier_shap,
    'Uniform ProxySPEX paired': uniform_proxyspex_paired,
    'Uniform ProxySPEX unpaired': uniform_proxyspex_unpaired,
    'Kernel ProxySPEX paired': kernel_proxyspex_paired,
    'Kernel ProxySPEX unpaired': kernel_proxyspex_unpaired,
    'SPEX SHAP': spex_shap,
}

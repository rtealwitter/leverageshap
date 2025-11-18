from .official_shap import *
from .leverage_shap import *
from .new import *

estimators = {
    'Kernel SHAP': official_kernel_shap,
    'Permutation SHAP': official_permutation_shap,
    'Leverage SHAP': leverage_shap,
    'Tree SHAP': official_tree_shap,
    'SPEX SHAP': spex_shap,
    # ProxySPEX (paired)
    'ProxySPEX Uniform Paired 1n': proxyspex_uniform_shap_paired_1n,
    'ProxySPEX Uniform Paired 2n': proxyspex_uniform_shap_paired_2n,
    'ProxySPEX Kernel Paired 1n': proxyspex_kernel_shap_paired_1n,
    'ProxySPEX Kernel Paired 2n': proxyspex_kernel_shap_paired_2n,
    # LASSO (paired)
    'LASSO Uniform Paired 1n': lasso_uniform_shap_paired_1n,
    'LASSO Uniform Paired 2n': lasso_uniform_shap_paired_2n,
    'LASSO Kernel Paired 1n': lasso_kernel_shap_paired_1n,
    'LASSO Kernel Paired 2n': lasso_kernel_shap_paired_2n,
    # ProxySPEX (unpaired)
    'ProxySPEX Uniform Unpaired 1n': proxyspex_uniform_shap_unpaired_1n,
    'ProxySPEX Uniform Unpaired 2n': proxyspex_uniform_shap_unpaired_2n,
    'ProxySPEX Kernel Unpaired 1n': proxyspex_kernel_shap_unpaired_1n,
    'ProxySPEX Kernel Unpaired 2n': proxyspex_kernel_shap_unpaired_2n,
    # LASSO (unpaired)
    'LASSO Uniform Unpaired 1n': lasso_uniform_shap_unpaired_1n,
    'LASSO Uniform Unpaired 2n': lasso_uniform_shap_unpaired_2n,
    'LASSO Kernel Unpaired 1n': lasso_kernel_shap_unpaired_1n,
    'LASSO Kernel Unpaired 2n': lasso_kernel_shap_unpaired_2n,
}

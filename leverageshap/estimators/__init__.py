from .official_shap import *
from .leverage_shap import *
from .new import *

estimators = {
    'Kernel SHAP': official_kernel_shap,
    'Permutation SHAP': official_permutation_shap,
    'Leverage SHAP': leverage_shap,
    'Tree SHAP': official_tree_shap,
    'SPEX SHAP': spex_shap,
    'ProxySPEX SHAP 2n': proxyspex_shap_2n,
    'ProxySPEX SHAP 4n': proxyspex_shap_4n,
    'ProxySPEX SHAP 6n': proxyspex_shap_6n,
    'ProxySPEX SHAP 8n': proxyspex_shap_8n,
}

from .official_shap import *
from .leverage_shap import *

estimators = {
    'Kernel SHAP': official_kernel_shap,
    'Permutation SHAP': official_permutation_shap,
    'Leverage SHAP': leverage_shap,
    'Tree SHAP': official_tree_shap,
}

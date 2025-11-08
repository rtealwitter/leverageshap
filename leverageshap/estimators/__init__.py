from .official_shap import *
from .leverage_shap import *
from .new import *

estimators = {
    'Kernel SHAP': official_kernel_shap,
    'Permutation SHAP': official_permutation_shap,
    'Leverage SHAP': leverage_shap,
    'Tree SHAP': official_tree_shap,
    'New SHAP': new_shap,
}

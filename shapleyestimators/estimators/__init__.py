import scipy.special
from .official_shap import *
from .regression import *

import numpy as np
import xgboost as xgb
import scipy

estimators = {
    'Official Kernel SHAP': official_kernel_shap,
    'Official Tree SHAP': official_tree_shap,
    'Kernel SHAP': kernel_shap,
    'Kernel SHAP Leverage' : kernel_shap_leverage,
    'Kernel SHAP Unpaired': kernel_shap_unpaired,
    'Kernel SHAP Leverage Unpaired': kernel_shap_leverage_unpaired,
    #'Weighted Regression': weighted_regression,
    #'Weighted Regression Leverage': weighted_regression_leverage,
}

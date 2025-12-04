import xgboost as xgb
import leverageshap as ls
import numpy as np

dataset = 'NHANES'
reps = 3
size_mults = [.25, .5, 1, 2, 4, 8, 16, 32, 64]

X, y = ls.load_dataset(dataset)
n = X.shape[1]

sample_sizes = [int(n * mult) for mult in size_mults]

# Collect all estimators exported by the package except 'Tree SHAP' (used as ground truth)
#estimator_names = [name for name in list(ls.estimators.keys()) if name not in ['Tree SHAP', 'Permutation SHAP']]
estimator_names = [
    'Regression MSR',
#    'Leverage SHAP',
#    'Fourier SHAP'
]

mse_by_estimator_and_sample_size = {
    name: {sample_size : [] for sample_size in sample_sizes} for name in estimator_names
}

for i in range(reps):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    baseline, explicand = ls.load_input(X, seed=i)

    true_shap = ls.estimators['Tree SHAP'](baseline, explicand, model, None).flatten()

    for estimator_name in estimator_names:
        estimator = ls.estimators[estimator_name]
        for sample_size in sample_sizes:            
            estimated_shap = estimator(baseline, explicand, model, sample_size).flatten()
            mse = np.mean((true_shap - estimated_shap)**2)

            mse_by_estimator_and_sample_size[estimator_name][sample_size].append(mse)

for estimator_name in estimator_names:
    for sample_size in sample_sizes:
        mses = mse_by_estimator_and_sample_size[estimator_name][sample_size]
        mean_mse = np.mean(mses)
        print(f'{dataset}, Estimator: {estimator_name}, n: {n}, m: {sample_size}, Mean MSE: {mean_mse:.3g}')


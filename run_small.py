import xgboost as xgb
import leverageshap as ls
import numpy as np

subtract_mobius1 = False

dataset = 'Communities'
reps = 3
size_mults = [2, 8, 64, 256]

X, y = ls.load_dataset(dataset)
n = X.shape[1]

sample_sizes = [int(n * mult) for mult in size_mults]

# Collect all estimators exported by the package except 'Tree SHAP' (used as ground truth)
#estimator_names = [name for name in list(ls.estimators.keys()) if name not in ['Tree SHAP', 'Permutation SHAP']]
estimator_names = [
    'Regression MSR',
    'Leverage SHAP',
#    'Fourier SHAP'
]

mse_by_estimator_and_sample_size = {
    name: {sample_size : [] for sample_size in sample_sizes} for name in estimator_names
}

mse_by_estimator_and_sample_size['Lev->Reg MSR'] = {sample_size : [] for sample_size in sample_sizes}

for i in range(reps):
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    baseline, explicand = ls.load_input(X, seed=i)

    true_shap = ls.estimators['Tree SHAP'](baseline, explicand, model, None).flatten()

    for estimator_name in estimator_names:
        estimator = ls.estimators[estimator_name]
        for sample_size in sample_sizes:            
            estimated_shap = estimator(baseline, explicand, model, sample_size, subtract_mobius1=subtract_mobius1).flatten()
            mse = np.mean((true_shap - estimated_shap)**2 / np.mean(true_shap**2))

            mse_by_estimator_and_sample_size[estimator_name][sample_size].append(mse)

    lev_estimator = ls.estimators['Leverage SHAP']
    regmsr_estimator = ls.estimators['Regression MSR']
    for sample_size in sample_sizes:
        lev_estimated_shap = lev_estimator(baseline, explicand, model, sample_size, subtract_mobius1=subtract_mobius1).flatten()
        regmsr_estimated_shap = regmsr_estimator(baseline, explicand, model, sample_size, subtract_mobius1=subtract_mobius1, estimated_phi=lev_estimated_shap).flatten()
        mse_lev_regmsr = np.mean((true_shap - regmsr_estimated_shap)**2 / np.mean(true_shap**2))
        mse_by_estimator_and_sample_size['Lev->Reg MSR'][sample_size].append(mse_lev_regmsr)

for estimator_name in mse_by_estimator_and_sample_size.keys():
    for sample_size in sample_sizes:
        mses = mse_by_estimator_and_sample_size[estimator_name][sample_size]
        mean_mse = np.mean(mses)
        print(f'{dataset}, Estimator: {estimator_name}, n: {n}, m: {sample_size}, Mean MSE: {mean_mse:.3g}')


import shap
import numpy as np

def official_kernel_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)

    explainer = shap.KernelExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True, l1_reg=False)
    return shap_values

def official_permutation_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    explicand = explicand.astype('float64')
    num_features = explicand.shape[1]
    # shap's PermutationExplainer spends 2n+1 evaluations per permutation and
    # raises ValueError when max_evals < 2n+1.  Pass the budget directly as
    # max_evals (the old `npermutations = m // n` route truncated to 0 for
    # m < n and was silently skipped by the benchmark); floor at 2n+1 with a
    # visible warning so small budgets are still recorded rather than dropped.
    min_evals = 2 * num_features + 1
    max_evals = num_samples
    if max_evals < min_evals:
        print(f'Warning: num_samples={num_samples} < 2n+1={min_evals} for Permutation SHAP (n={num_features}); using {min_evals} evaluations instead.')
        max_evals = min_evals

    explainer = shap.PermutationExplainer(eval_model, baseline)
    explanation = explainer(explicand, max_evals=max_evals, silent=True)
    return explanation.values

def official_shapley_sampling(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    explainer = shap.SamplingExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True)
    return shap_values

def official_tree_shap(baseline, explicand, model, num_samples):
    # Suppress warning only for this function
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(model, baseline)
    shap_values = explainer.shap_values(explicand)
    # Re-enable warnings
    warnings.filterwarnings("default", category=UserWarning)
    return shap_values
import xgboost as xgb
import leverageshap as ls
import numpy as np

def run_shap_experiment(dataset, size_mults, estimator_names, seed=42, num_reps=10, verbose=False):
    # Load dataset
    X, y = ls.load_dataset(dataset)
    n = X.shape[1]
    
    # Calculate exact sample sizes based on multipliers
    sample_sizes = [int(n * mult) for mult in size_mults]

    if verbose:
        print(f"Parameters: dataset={dataset}, n={n}, seed={seed}, num_reps={num_reps}\n")

    # Initialize performance tracking dict: estimator -> sample_size -> [mses]
    performance = {
        name: {sample_size: [] for sample_size in sample_sizes} 
        for name in estimator_names
    }

    for rep in range(num_reps):
        # Generate a distinct, reproducible seed for this repetition
        seed_rep = int(seed + 1234567 * rep)
        
        # Fit model (passed random_state for full reproducibility)
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed_rep)
        model.fit(X, y)

        # Get inputs for this repetition using the rep seed
        baseline, explicand = ls.load_input(X, seed=seed_rep)

        # Compute ground truth using Tree SHAP
        true_shap = ls.estimators['Tree SHAP'](baseline, explicand, model, None).flatten()

        for estimator_name in estimator_names:
            estimator = ls.estimators[estimator_name]
            
            for sample_size in sample_sizes:
                # Run the mechanism
                estimated_shap = estimator(baseline, explicand, model, sample_size).flatten()

                # Calculate standard MSE
                mse = np.mean((true_shap - estimated_shap)**2) / np.mean(true_shap**2)
                
                # Store the result
                performance[estimator_name][sample_size].append(mse)

    if verbose:
        # Build dynamic header based on the estimators provided
        header = f"{'Sample Size':<12s}"
        for est in estimator_names:
            col_name = f"{est} (Avg ± Std [Med])"
            header += f" | {col_name:<30s}"
            
        print(f"\n{header}")
        print("-" * len(header))
        
        # Print a row for each sample size
        for sample_size in sample_sizes:
            row_str = f"{sample_size:<12d}"
            for est in estimator_names:
                mses = performance[est][sample_size]
                avg_mse = np.mean(mses)
                std_mse = np.std(mses)
                med_mse = np.median(mses)
                
                # Combine metrics into a single readable string per cell
                cell_str = f"{avg_mse:.1e} ± {std_mse:.1e} [{med_mse:.1e}]"
                row_str += f" | {cell_str:<30s}"
            print(row_str)

    return performance


estimator_names = ['Kernel SHAP', 'Leverage SHAP', 'SNGD']
size_mults = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

performance = run_shap_experiment(
    dataset='NHANES',
    size_mults=size_mults,
    estimator_names=estimator_names,
    seed=42,
    num_reps=10,
    verbose=True
)
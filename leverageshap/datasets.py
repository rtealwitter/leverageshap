import shap
import numpy as np

dataset_loaders = {
    'Adult' : shap.datasets.adult,
    'California' : shap.datasets.california,
    'Communities' : shap.datasets.communitiesandcrime,
    'Correlated' : shap.datasets.corrgroups60,
    'Diabetes' : shap.datasets.diabetes,
    'Independent' : shap.datasets.independentlinear60,
    'IRIS' : shap.datasets.iris,
    'NHANES' : shap.datasets.nhanesi,
}

def load_dataset(dataset_name):
    X, y = dataset_loaders[dataset_name]()
    # Remove nan values
    X = X.fillna(X.mean())
    return X, y

def load_input(X, seed=None):
    if seed is not None:
        np.random.seed(seed)
    baseline = X.mean().values.reshape(1, -1)
    explicand_idx = np.random.choice(X.shape[0])
    explicand = X.iloc[explicand_idx].values.reshape(1, -1)
    for i in range(explicand.shape[1]):
        while baseline[0, i] == explicand[0, i]:
            explicand_idx = np.random.choice(X.shape[0])
            explicand[0,i] = X.iloc[explicand_idx, i]
    return baseline, explicand
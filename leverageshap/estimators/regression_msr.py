import numpy as np
from .sampling import CoalitionSampler
from .helpers import Game
from scipy.special import comb as binom

from xgboost import XGBRegressor
import shap

class RegressionMSR:
    def __init__(self, n, game, paired_sampling=True, random_state=42):
        self.game = game
        self.n = n
        self.paired_sampling = paired_sampling
        self.random_state = random_state
    
    def shap_values(self, num_samples):
        if num_samples < 6:
            print('Number of samples too small, setting to 6')
            num_samples = 6
        
        sizes = np.arange(self.n)
        shapley_weights_by_size = 1 / (self.n * binom(self.n-1, sizes))

        sampling_weights = np.ones(self.n-1)

        sampler = CoalitionSampler(
            n_players = self.n,
            sampling_weights = sampling_weights,
            pairing_trick = self.paired_sampling,
            random_state = self.random_state
        )

        sampler.sample(num_samples)
        coalitions_matrix = sampler.coalitions_matrix
        coalitions_probability = sampler.coalitions_probability
        coalitions_size = sampler.coalitions_size

        game_values = self.game(coalitions_matrix)

        model = XGBRegressor(random_state=self.random_state)
        model.fit(coalitions_matrix, game_values)

        explainer = shap.TreeExplainer(
            model, feature_perturbation="interventional", data=np.zeros((1, self.n))
        )

        tree_phi = explainer.shap_values(np.ones(self.n))
        tree_values = model.predict(coalitions_matrix)
        residual_values = game_values - tree_values

        phi = np.zeros(self.n)

        for idx in range(self.n):
            idx_contained = (coalitions_matrix[:, idx] == 1)
            not_contained = ~idx_contained
            mean_with = mean_without = 0
            if np.any(idx_contained):
                mean_with = (
                    residual_values[idx_contained] * shapley_weights_by_size[coalitions_size[idx_contained]-1]
                    / coalitions_probability[idx_contained]
                ).mean()
            if np.any(not_contained):
                mean_without = (
                    residual_values[not_contained] * shapley_weights_by_size[coalitions_size[not_contained]]
                    / coalitions_probability[not_contained]
                ).mean()
            phi[idx] += tree_phi[idx] + mean_with - mean_without

        return phi

def regression_msr(baseline, explicand, model, num_samples):
    game = Game(model, baseline, explicand)
    n = baseline.shape[1]
    estimator = RegressionMSR(n, game, paired_sampling=True)
    return estimator.shap_values(num_samples)

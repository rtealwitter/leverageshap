import numpy as np
import scipy
from ..utils import *
import scipy.special

# Compute b
# Compute A
# Compute v1-v0

class RegressionEstimator:
    def __init__(self, model, baseline, explicand, num_samples, paired_sampling=False, leverage_sampling=False, bernoulli_sampling=False):
        self.model = model
        self.baseline = baseline
        self.explicand = explicand
        # Subtract 2 for the baseline and explicand and ensure num_samples is even
        self.num_samples = int((num_samples -2 ) // 2) * 2
        self.paired_sampling = paired_sampling
        self.n = self.baseline.shape[1] # Number of features
        self.gen = np.random.Generator(np.random.PCG64())
        self.sample_weight = lambda s : 1 / (s * (self.n - s)) if not leverage_sampling else np.ones_like(s)
        self.reweight = lambda s : 1 / (self.sample_weight(s) * (s * (self.n - s)))
        self.kernel_weights = []
        self.sample = self.sample_with_replacement if not bernoulli_sampling else self.sample_without_replacement
        #self.used_indices = set()
    
    def add_one_sample(self, idx, indices, weight):
        #indices = sorted(indices)
        #if tuple(indices) in self.used_indices: return
        #self.used_indices.add(tuple(indices))
        if not self.paired_sampling:
            self.SZ_binary[idx, indices] = 1
            self.kernel_weights.append(weight)
        else:
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self.SZ_binary[2*idx, indices] = 1
            self.kernel_weights.append(weight)
            self.SZ_binary[2*idx+1, indices_complement] = 1
            self.kernel_weights.append(weight)

    
    def sample_with_replacement(self):
        self.SZ_binary = np.zeros((self.num_samples, self.n))
        valid_sizes = np.array(list(range(1, self.n)))
        prob_sizes = self.sample_weight(valid_sizes)
        prob_sizes = prob_sizes / np.sum(prob_sizes)
        num_sizes = self.num_samples if not self.paired_sampling else self.num_samples // 2
        sampled_sizes = self.gen.choice(valid_sizes, num_sizes, p=prob_sizes)
        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False)
            # weight = Pr(sampling this set) * w(s)
            weight = 1 / (self.sample_weight(s) * s * (self.n - s))
            self.add_one_sample(idx, indices, weight=weight)
    
    def find_constant_for_bernoulli(self, max_C = 1e10):
        # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        C = 1 # Assume at least n - 1 samples
        m = min(self.num_samples, 2**self.n-2) # Maximum number of samples is 2^n -2
        def expected_samples(C):
            expected = [min(scipy.special.binom(self.n, s), C * self.sample_weight(s)) for s in range(1, self.n)]
            #print(f'Expected samples: {np.sum(expected)}')
            #print(f'Constraint: {m}')
            #print(f'C: {C}')
            return np.sum(expected)
        # Efficiently find C with binary search
        L = 1
        R = scipy.special.binom(self.n, self.n // 2)
        while round(expected_samples(C)) != m:
            if expected_samples(C) < m: L = C
            else: R = C
            C = (L + R) / 2
        self.C = round(C)
    
    def sample_without_replacement(self):
        self.find_constant_for_bernoulli()
        m_s_all = []
        for s in range(1, self.n):
            # Sample from Binomial distribution with (n choose s) trials and probability min(1, C*sample_weight(s) / (n choose s))
            prob = min(1, self.C * self.sample_weight(s) / scipy.special.binom(self.n, s))
            try:
                m_s = self.gen.binomial(int(scipy.special.binom(self.n, s)), prob)
            except OverflowError: # If the number of samples is too large, assume the number of samples is the expected number
                m_s = int(prob * scipy.special.binom(self.n, s))
            if self.paired_sampling:
                if s == self.n // 2: # Already sampled all larger sets with the complement                    
                    if self.n % 2 == 0: # Special handling for middle set size if n is even
                        m_s_all.append(m_s // 2)
                    else: m_s_all.append(m_s)
                    break
            m_s_all.append(m_s)
        sampled_m = np.sum(m_s_all)
        num_rows = sampled_m if not self.paired_sampling else sampled_m * 2
        self.SZ_binary = np.zeros((num_rows, self.n))
        idx = 0
        for s, m_s in enumerate(m_s_all):
            s += 1
            prob = min(1, self.C * self.sample_weight(s) / scipy.special.binom(self.n, s))
            weight = 1 / (prob * scipy.special.binom(self.n, s) * (self.n - s) * s )
            if self.paired_sampling and s == self.n // 2 and self.n % 2 == 0:
                # Partition the all middle sets into two
                # based on whether the combination contains n-1
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices) + [self.n-1], weight = weight)
                    idx += 1
            else:
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices), weight = weight)
                    idx += 1
    
    def compute(self):
        # Sample
        self.sample()
        # A = Z P
        # y = v(z) - v0
        # b = y - Z1 (v1 - v0) / n    
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n

        # Remove zero rows
        SZ_binary = self.SZ_binary[np.sum(self.SZ_binary, axis=1) != 0]
        v0, v1 = self.model.predict(self.baseline), self.model.predict(self.explicand)
        inputs = self.baseline * (1 - SZ_binary) + self.explicand * SZ_binary
        Sy = self.model.predict(inputs) - v0
        SZ_binary1 = np.sum(SZ_binary, axis=1)
        
        Sb = Sy - (v1 - v0) * SZ_binary1 / self.n

        # Projection matrix
        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        PZSSb = P @ SZ_binary.T @ np.diag(self.kernel_weights) @ Sb
        PZSSZP = P @ SZ_binary.T @ np.diag(self.kernel_weights) @ SZ_binary @ P
        PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

        self.phi = PZSSZP_inv_PZSSb + (v1 - v0) / self.n

        return self.phi

def leverage_shap(baseline, explicand, model, num_samples, paired_sampling=False, bernoulli_sampling=False):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=paired_sampling, leverage_sampling=True, bernoulli_sampling=bernoulli_sampling)
    return estimator.compute()

def leverage_shap_paired(baseline, explicand, model, num_samples):
    return leverage_shap(baseline, explicand, model, num_samples, paired_sampling=True)

def leverage_shap_bernoulli(baseline, explicand, model, num_samples):
    return leverage_shap(baseline, explicand, model, num_samples, paired_sampling=True, bernoulli_sampling=True)

def kernel_shap(baseline, explicand, model, num_samples, paired_sampling=False):
    estimator = RegressionEstimator(model, baseline, explicand, num_samples, paired_sampling=paired_sampling, leverage_sampling=False)
    return estimator.compute()

def kernel_shap_paired(baseline, explicand, model, num_samples):
    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=True)

#def leverage_shap_new(baseline, explicand, model, num_samples, paired_sampling=False):
#    num_samples -= 2 # Subtract 2 for the baseline and explicand
#    num_samples = int((num_samples) // 2) * 2 # Make sure num_samples is even
#    eval_model = lambda z : model.predict(explicand * z + baseline * (1 - z))
#    num_features = baseline.shape[1]
#    gen = np.random.Generator(np.random.PCG64())
#    v0 = eval_model(np.zeros(num_features))
#    v1 = eval_model(np.ones(num_features))
#
#    SZ_binary = np.zeros((num_samples, num_features))
#    
#    valid_sizes = np.array(list(range(1, num_features)))
#    prob_sizes = np.ones_like(valid_sizes) / len(valid_sizes)
#    sampled_sizes = gen.choice(valid_sizes, num_samples, p=prob_sizes)
#    for idx, s in enumerate(sampled_sizes):
#        indices = gen.choice(num_features, s, replace=False)
#        if not paired_sampling:
#            SZ_binary[idx, indices] = 1
#        if paired_sampling: # Add complement if paired sampling
#            indices_complement = np.array([i for i in range(num_features) if i not in indices])
#            SZ_binary[2*idx, indices_complement] = 1
#            SZ_binary[2*idx+1, indices] = 1
#            # Break half way through if paired sampling
#            if idx >= (num_samples - 2) // 2 -1: break
#    
#    # A = Z P
#    # y = v(z) - v0
#    # b = y - Z1 (v1 - v0) / n    
#    # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
#    # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n
#
#    # Remove zero rows
#    SZ_binary = SZ_binary[np.sum(SZ_binary, axis=1) != 0]
#
#    inputs = baseline * (1 - SZ_binary) + explicand * SZ_binary
#    Sy = model.predict(inputs) - v0
#    SZ_binary1 = np.sum(SZ_binary, axis=1)
#    b = Sy - (v1 - v0) * SZ_binary1 / num_features
#
#    # Projection matrix
#    P = np.eye(num_features) - 1/num_features * np.ones((num_features, num_features))
#
#    weight = lambda s : 1 / (s * (num_features - s))
#
#    PZSSb = P @ SZ_binary.T @ np.diag(weight(SZ_binary1)) @ b
#    
#    PZSSZP = P @ SZ_binary.T @ np.diag(weight(SZ_binary1)) @ SZ_binary @ P
#    PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]
#    phi = PZSSZP_inv_PZSSb + (v1 - v0) / num_features
#
#    return phi
#
#def leverage_shap_new_paired(baseline, explicand, model, num_samples):
#    return leverage_shap_new(baseline, explicand, model, num_samples, paired_sampling=True)
#
#def get_components(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False):
#    num_samples -= 2 # Subtract 2 for the baseline and explicand
#    num_samples = int((num_samples) // 2) * 2 # Make sure num_samples is even
#    eval_model = lambda z : model.predict(explicand * z + baseline * (1 - z))
#    num_features = baseline.shape[1]
#    gen = np.random.Generator(np.random.PCG64())
#    v0 = eval_model(np.zeros(num_features))
#    v1 = eval_model(np.ones(num_features))
#    
#    # Real weights
#    compute_weights = lambda s: 1 / (s * (num_features - s))
#    # Select weights to sample each size
#    distribution = lambda s: 1 / (s * (num_features - s)) if not leverage_sampling else np.ones_like(s)
#
#    valid_sizes = np.array(list(range(1, num_features)))
#    prob_s = distribution(valid_sizes)
#
#    Z = np.zeros((num_samples, num_features))
#    
#    prob_s = prob_s / np.sum(prob_s)
#    sampled_sizes = gen.choice(valid_sizes, num_samples-2, p=prob_s)
#    for idx, s in enumerate(sampled_sizes):
#        indices = gen.choice(num_features, s, replace=False)
#        if not paired_sampling:
#            Z[idx, indices] = 1
#        if paired_sampling: # Add complement if paired sampling        
#            indices_complement = np.array([i for i in range(num_features) if i not in indices])
#            Z[2*idx, indices_complement] = 1
#            Z[2*idx+1, indices] = 1
#            # Break half way through if paired sampling
#            if idx >= (num_samples - 2) // 2 -1: break
#    Z1_norm = np.sum(Z, axis=1)
#
#    # Remove zero rows
#    Z = Z[Z1_norm != 0]
#    Z1_norm = Z1_norm[Z1_norm != 0]
#
#    reweighting = compute_weights(Z1_norm) / distribution(Z1_norm)
#
#    inputs = baseline * (1 - Z) + explicand * Z
#    vz = model.predict(inputs)
#
#    ZTZ = Z.T @ np.diag(reweighting) @ Z
#    ZTv = Z.T @ np.diag(reweighting) @ (vz - v0)
#    
#    assert ZTZ.shape == (num_features, num_features)
#    assert ZTv.shape == (num_features,)
#
#    return {
#        'ZTZ': ZTZ,
#        'ZTv': ZTv,
#        'delta': v1 - v0,
#    }
#
#def kernel_shap(baseline, explicand, model, num_samples, paired_sampling=False, leverage_sampling=False):
#    components = get_components(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=leverage_sampling)
#    ZTZ = components['ZTZ']
#    ZTv = components['ZTv']
#    delta = components['delta']
#    ZTZ_inv_ones = np.linalg.solve(ZTZ, np.ones_like(ZTv))
#    ZTZ_inv_ZTv = np.linalg.solve(ZTZ, ZTv)
#
#    return (
#        ZTZ_inv_ZTv -
#        ZTZ_inv_ones * (
#            np.sum(ZTZ_inv_ZTv) - delta
#        ) / np.sum(ZTZ_inv_ones)
#    )
#
#def kernel_shap_paired(baseline, explicand, model, num_samples):
#    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=False)
#
#def kernel_shap_leverage(baseline, explicand, model, num_samples, paired_sampling=False):
#    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=paired_sampling, leverage_sampling=True)
#
#def kernel_shap_leverage_paired(baseline, explicand, model, num_samples):
#    return kernel_shap(baseline, explicand, model, num_samples, paired_sampling=True, leverage_sampling=True)
#
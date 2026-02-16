import numpy as np

class Game:
    def __init__(self, model, baseline, explicand, subtract_mobius1=False, estimated_phi=None):
        self.model = model
        self.baseline = baseline
        self.explicand = explicand
        self.subtract_mobius1 = subtract_mobius1            
        self.estimated_phi = estimated_phi
        self.zero, self.singletons = self.compute_mobius1()
    
    def compute_mobius1(self):
        # Compute the Mobius transform of order 1
        zero_val = self.model.predict(self.baseline)
        singletons = np.zeros(self.baseline.shape[1])
        for j in range(self.baseline.shape[1]):
            subset_binary = np.zeros(self.baseline.shape[1])
            subset_binary[j] = 1
            one_val = self.__call__(subset_binary.reshape(1, -1), insetup=True) - zero_val
            singletons[j] = one_val
        return zero_val, singletons
    
    def __call__(self, S, insetup=False):
        # S is a m by n binary matrix
        inputs = self.baseline * (1 - S) + self.explicand * S
        raw_val = self.model.predict(inputs)
        if self.estimated_phi is not None and not insetup:
            return raw_val - np.dot(S, self.estimated_phi) - self.zero
        if self.subtract_mobius1 and not insetup:
            # Subtract the Mobius transform of order 1
            singleton_contributions = np.dot(S, self.singletons)
            return raw_val - singleton_contributions - self.zero
        if not insetup:
            return raw_val - self.zero
        return raw_val
    
    def edge_cases(self):
        v0 = self.model.predict(self.baseline)
        v1 = self.model.predict(self.explicand)
        return v0, v1
    
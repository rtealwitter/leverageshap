class Game:
    def __init__(self, model, baseline, explicand):
        self.model = model
        self.baseline = baseline
        self.explicand = explicand
    
    def __call__(self, S):
        # S is a m by n binary matrix
        inputs = self.baseline * (1 - S) + self.explicand * S
        return self.model.predict(inputs)
    
    def edge_cases(self):
        v0 = self.model.predict(self.baseline)
        v1 = self.model.predict(self.explicand)
        return v0, v1
    
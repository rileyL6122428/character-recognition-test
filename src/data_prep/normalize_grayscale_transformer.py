from data_management.adapters import normalize_image_0_to_1

class NormalizeGrayscaleTransformer:

    def __init__(self, **params):
        self.set_params(params)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return normalize_image_0_to_1(X) if self.is_on else X
    
    def set_params(self, is_on=True):
        self.is_on = is_on

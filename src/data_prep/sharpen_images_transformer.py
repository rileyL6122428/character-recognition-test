import numpy

class SharpenImageTransformer:

    def __init__(self, whiten_threshold=None, darken_threshold=None):
        self.whiten_threshold = whiten_threshold
        self.darken_threshold = darken_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        fitted = []

        for flat_image in X:
            next_row = []
            for grayscale_val in flat_image:
                if self.whiten_threshold is not None and grayscale_val <= self.whiten_threshold:
                    next_row.append(0)
                elif self.darken_threshold is not None and grayscale_val >= self.darken_threshold:
                    next_row.append(255)
                else:
                    next_row.append(grayscale_val)
            fitted.append(next_row)

        return numpy.array(fitted)

    def set_params(self, whiten_threshold=None, darken_threshold=None):
        self.whiten_threshold = whiten_threshold
        self.darken_threshold = darken_threshold


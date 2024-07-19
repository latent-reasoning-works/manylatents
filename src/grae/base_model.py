class BaseModel:
    def __init__(self):
        """Init."""

    def fit(self, x):
        raise NotImplementedError()

    def fit_transform(self, x):

        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        raise NotImplementedError()

    def inverse_transform(self, x):
        raise NotImplementedError()

    def reconstruct(self, x):
        return self.inverse_transform(self.transform(x))

class AE(BaseModel):

    def __init__(self):
        super().__init__()


import os
import joblib

class Model:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def path(self, model_path):
        self.model_path = model_path
        return self.model_path

    def load(self):
        if self.model_path is None:
            raise ValueError('Model path not specified')

        elif not os.path.exists(self.model_path):
            raise FileNotFoundError(f'File {self.model_path} not found')

        self.model = joblib.load(self.model_path)

    def predict(self, X):
        if self.model is None:
            raise ValueError('Model not loaded')
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError('Model not loaded')
        return self.model.predict_proba(X)

    def __str__(self):
        return str(self.model)


if __name__ == '__main__':
    model_fallecidos = Model('models/stacking_classifier_fallecidos.pkl')
    model_fallecidos.load()
    print(model_fallecidos.predict([[10, 2, 3, 4, 5, 6, 7, 10]]))
    print(model_fallecidos.predict_proba([[1, 2, 3, 4, 5, 6, 7, 8]]))
    print(model_fallecidos)

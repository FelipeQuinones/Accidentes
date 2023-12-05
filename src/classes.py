import os
import pandas as pd
import joblib

class Model:
    def __init__(self, model_path=None, model=None):
        if model_path is not None:
            self.load(model_path)
        else:
            self.model = model

    def load(self, model_path=None):
        if model_path is None:
            raise ValueError('Model path not specified')
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f'File {model_path} not found')
        self.model = joblib.load(model_path)
        print(f'Model {model_path} loaded')

    def predict(self, X):
        if self.model is None:
            raise ValueError('Model not loaded')
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError('Model not loaded')
        return self.model.predict_proba(X)

    def __str__(self):
        return '-'*75 + '\n' + str(self.model) + '\n' + '-'*75
    
class Columns:
    def __init__(self, column_path=None, columns=None):
        if column_path is not None:
            self.load(column_path)
        else:
            self.columns = columns

    def load(self, csv_path):
        if csv_path is None:
            raise ValueError('CSV path not specified')
        elif not os.path.exists(csv_path):
            raise FileNotFoundError(f'File {csv_path} not found')
        self.columns = pd.read_csv(csv_path, index_col=0)
        return self.columns

    def scaler(self, X):
        if self.columns is None:
            raise ValueError('Columns not loaded')
        return [(x - column['Min']) / (column['Max'] - column['Min']) for x, column in zip(X[0], self.columns.to_dict('records'))]

    def indexes(self):
        if self.columns is None:
            raise ValueError('Columns not loaded')
        return self.columns.index.to_list()

    def __str__(self):
        return '-'*25 + '\n' + str(self.columns) + '\n' + '-'*25
    
    def __repr__(self):
        return str(self.columns)

if __name__ == '__main__':
    model_fallecidos = Model()
    model_fallecidos.load('models/stacking_classifier_fallecidos.pkl')
    print(model_fallecidos.predict([[6, 2, 11, 11, 5, 6, 7, 10]]))
    print(model_fallecidos.predict_proba([[6, 2, 11, 11, 5, 6, 7, 10]]))
    print(model_fallecidos)

    columns = Columns()
    columns.load('data/min_max_columns.csv')
    print(columns,'\n')
    print(columns.scaler([[6, 2, 11, 11, 5, 6, 7, 10]]))
    print(columns.indexes())

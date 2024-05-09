from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class ReplaceOutliers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower_bounds = X.quantile(0.25) - 1.25 * (X.quantile(0.75) - X.quantile(0.25))
        self.upper_bounds = X.quantile(0.75) + 1.25 * (X.quantile(0.75) - X.quantile(0.25))
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for column in X_copy.columns:
            lower_bound = self.lower_bounds[column]
            upper_bound = self.upper_bounds[column]
            X_copy[column] = X_copy[column].clip(lower=lower_bound, upper=upper_bound)
        return X_copy

# Charger le modèle
model = joblib.load('Wine_quality_classification.pkl')

app = Flask(__name__)

def preprocess(data):

    data_df = pd.DataFrame(data, index=[0])

    preparation_pipeline = Pipeline([
        ('outlier_replacer', ReplaceOutliers()),
        ('scaler', StandardScaler())
    ])

        # Apply the pipeline
    preparation_pipeline.fit(data_df)
    data_prepared = preparation_pipeline.transform(data_df)

    return data_prepared

@app.route('/')
def index():
    return 'Welcome to the Wine Quality Prediction API'

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json['data']
    
    data = preprocess(data)
    
    #predict
    prediction = model.predict(np.array(data).reshape(1, -1))
    
    # return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()
import requests
import json

# URL
url = 'http://127.0.0.1:5000/predict'

# Data to predict, return 6 in my case
data = {
    'data': {
        'fixed acidity': 7.8,
        'volatile acidity': 0.565,
        'citric acid': 0.4,
        'residual sugar': 1.9,
        'chlorides': 0.07,
        'free sulfur dioxide': 50,
        'total sulfur dioxide': 16,
        'density': 0.9965,
        'pH': 3.0,
        'sulphates': 0.6,
        'alcohol': 11
    }
}

# Envoi de la requête POST
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    result = response.json()
    print("Prédiction:", result['prediction'])
else:
    print("Erreur:", response.text)


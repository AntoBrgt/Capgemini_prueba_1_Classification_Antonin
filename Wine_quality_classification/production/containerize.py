import os
import joblib
from sklearn.ensemble import VotingClassifier

#Load model
model = joblib.load(os.path.join('Wine_quality_classification.pkl'))

# Create model class

class WineQualityClassifier:
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        return self.model.predict(X)
    
# Init class
wine_classifier = WineQualityClassifier(model)

# Save it
joblib.dump(wine_classifier, 'Wine_quality_classification\production\wine_classifier.pkl')

# Execute docker API

import os
os.chdir('Wine_quality_classification\production')

os.system('docker build -t wine_classifier_api .')

os.system('docker run -p 5000:5000 wine_classifier_api')
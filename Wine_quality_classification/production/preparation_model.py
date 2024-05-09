import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#Load datas

wine_data = pd.read_csv(os.path.join("Wine_quality_classification", "data", 'WineQT.csv'))

#Remove the Id columns because it's not used
wine_data = wine_data.drop(columns=['Id'])

#Split in train and test

wine_data_feature = wine_data.drop('quality', axis=1)
wine_data_label = wine_data['quality']

wine_data_train, wine_data_test, wine_data_train_label, wine_data_test_label = train_test_split(wine_data_feature, wine_data_label, test_size=0.1, stratify=wine_data_label, random_state=42)

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

#Prepare datas

wine_data_train_prepared = wine_data_train.copy()

preparation_pipeline = Pipeline([
    ('outlier_replacer', ReplaceOutliers()),
    ('scaler', StandardScaler())
])

# Apply the pipeline
preparation_pipeline.fit(wine_data_train_prepared, wine_data_train_label)
wine_data_train_prepared = preparation_pipeline.transform(wine_data_train_prepared)

# Train model
decision_tree = DecisionTreeClassifier(ccp_alpha=0, class_weight=None, max_depth=None, max_features='sqrt',max_leaf_nodes= 10, min_samples_leaf = 5, min_samples_split = 2, min_weight_fraction_leaf = 0.0) 

decision_tree.fit(wine_data_train_prepared, wine_data_train_label)

random_forest = RandomForestClassifier(bootstrap=False, max_depth=None, max_features='sqrt', min_samples_leaf=3, min_samples_split=2, min_weight_fraction_leaf=0.1, n_estimators=200)
random_forest.fit(wine_data_train_prepared, wine_data_train_label)

logistic_regression = LogisticRegression(C=0.1, penalty='l2')
logistic_regression.fit(wine_data_train_prepared, wine_data_train_label)

svc = SVC(C = 0.1, coef0=1.0, degree=5, gamma='scale', kernel = 'poly')
svc.fit(wine_data_train_prepared, wine_data_train_label)

models = [
    ('logistic_regression', logistic_regression),
    ('random_forest', random_forest),
    ('decision_tree', decision_tree),
    ('SVC', svc)
]


voting_classifier = VotingClassifier(estimators=models, voting='hard')
voting_classifier.fit(wine_data_train_prepared, wine_data_train_label)


preparation_pipeline.fit(wine_data_test, wine_data_test_label)
wine_data_test_prepared = preparation_pipeline.transform(wine_data_test)


# Evaluate model
accuracy = voting_classifier.score(wine_data_test_prepared, wine_data_test_label)
print("Accuracy:", accuracy)

# save model
joblib.dump(voting_classifier, 'Wine_quality_classification\production\Wine_quality_classification.pkl')
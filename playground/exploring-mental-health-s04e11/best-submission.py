'''Ensemble model cross-validation scores: [0.9376688  0.93997868 0.93756219 0.94008529 0.94076048]
Mean cross-validation score: 0.9392
Optimized submission file created successfully!

lb score: 0.94195

'''

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Data Preprocessing
# Separate numeric and categorical columns
numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns.drop(['id', 'Depression'])
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Impute missing values
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

train_data[numeric_cols] = numeric_imputer.fit_transform(train_data[numeric_cols])
train_data[categorical_cols] = categorical_imputer.fit_transform(train_data[categorical_cols])

test_data[numeric_cols] = numeric_imputer.transform(test_data[numeric_cols])
test_data[categorical_cols] = categorical_imputer.transform(test_data[categorical_cols])

# Label Encoding with handling for unseen labels
for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = test_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Scale numeric features
scaler = StandardScaler()
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

# Separate features and target variable
X = train_data.drop(columns=['id', 'Depression'])
y = train_data['Depression']
X_test = test_data.drop(columns=['id'])

# Define individual models
lgb = LGBMClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
extra_trees = ExtraTreesClassifier(random_state=42)

# Train individual models
lgb.fit(X, y)
xgb.fit(X, y)
extra_trees.fit(X, y)

# Predictions for weight optimization
lgb_preds = lgb.predict_proba(X)[:, 1]
xgb_preds = xgb.predict_proba(X)[:, 1]
extra_trees_preds = extra_trees.predict_proba(X)[:, 1]

# Optimize weights using GridSearchCV
def voting_score(weights):
    final_preds = (
        weights[0] * lgb_preds +
        weights[1] * xgb_preds +
        weights[2] * extra_trees_preds
    )
    final_preds = (final_preds >= 0.5).astype(int)
    return accuracy_score(y, final_preds)

from scipy.optimize import minimize

# Objective function to minimize negative accuracy
def objective(weights):
    return -voting_score(weights)

# Constraints and bounds for weights
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
bounds = [(0, 1)] * 3

# Find optimal weights
initial_weights = [1/3, 1/3, 1/3]
result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
optimal_weights = result.x
print("Optimal Weights:", optimal_weights)

# Voting Classifier with optimized weights
ensemble_model = VotingClassifier(
    estimators=[
        ('LightGBM', lgb),
        ('XGBoost', xgb),
        ('ExtraTrees', extra_trees)
    ],
    voting='soft',
    weights=optimal_weights
)

# Train ensemble model
ensemble_model.fit(X, y)

# Cross-validation score for the ensemble model
cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='accuracy')
print(f"Ensemble model cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.4f}")

# Final predictions on test data
test_predictions = ensemble_model.predict(X_test)

# Prepare submission file
submission = sample_submission.copy()
submission['Depression'] = test_predictions
submission.to_csv('best-submission.csv', index=False)
print("Optimized submission file created successfully!")

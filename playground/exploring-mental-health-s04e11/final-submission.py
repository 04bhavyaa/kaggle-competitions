'''Ensemble model cross-validation scores: [0.93830846 0.94008529 0.93756219 0.93987207 0.94083156]
Mean cross-validation score: 0.9393
Optimized submission file created successfully!

lb score: 0.94232'''

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
    # Transform test data, assigning -1 for unseen labels
    test_data[col] = test_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Scale numeric features
scaler = StandardScaler()
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])

# Separate features and target variable
X = train_data.drop(columns=['id', 'Depression'])
y = train_data['Depression']
X_test = test_data.drop(columns=['id'])

# Hyperparameter tuning for LightGBM
lgb_params = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'max_depth': [-1, 10, 20],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgb = LGBMClassifier(random_state=42)
grid_search = GridSearchCV(estimator=lgb, param_grid=lgb_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Best LightGBM model after hyperparameter tuning
best_lgb = grid_search.best_estimator_
print("Best LightGBM Parameters:", grid_search.best_params_)
print("Best LightGBM CV Accuracy:", grid_search.best_score_)

# Train XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X, y)

# Ensembling with Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('LightGBM', best_lgb),
    ('XGBoost', xgb)
], voting='soft')
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
submission.to_csv('final-submission.csv', index=False)
print("Optimized submission file created successfully!")

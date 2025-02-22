import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Load the datasets
train = pd.read_csv('/Users/willarsenault/PycharmProjects/DS310P1/venv/train.csv')
x_test = pd.read_csv('/Users/willarsenault/PycharmProjects/DS310P1/venv/x_test.csv')

# Define feature columns (assuming they are labeled 'Col 1' to 'Col 64')
feature_cols = [f'Col {i}' for i in range(1, 65)]

# Separate features and target from the training data
X = train[feature_cols]
y = train['y']

# Split into training and validation sets (80/20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up a pipeline including:
# - Polynomial interaction features (degree set as a hyperparameter)
# - Scaler (Standard or Robust)
# - A generic regressor placeholder
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('scaler', StandardScaler()),
    ('pca', 'passthrough'),
    ('regressor', ElasticNet(max_iter=10000))
])

# Create an expanded parameter grid including:
# - Varying polynomial degree
# - Different scalers
# - Optional PCA
# - Linear models with extended hyperparameters
# - Ensemble methods as alternative regressors
param_grid = [
    # Linear Models: ElasticNet, Lasso, Ridge
    {
        'poly__degree': [2, 3],
        'scaler': [StandardScaler(), RobustScaler()],
        'pca': [PCA(n_components=50), 'passthrough'],
        'regressor': [ElasticNet(max_iter=10000)],
        'regressor__alpha': [0.1, 1, 10, 50, 100],
        'regressor__l1_ratio': [0.1, 0.5, 0.9]
    },
    {
        'poly__degree': [2, 3],
        'scaler': [StandardScaler(), RobustScaler()],
        'pca': [PCA(n_components=50), 'passthrough'],
        'regressor': [Lasso(max_iter=10000)],
        'regressor__alpha': [0.1, 1, 10, 50, 100]
    },
    {
        'poly__degree': [2, 3],
        'scaler': [StandardScaler(), RobustScaler()],
        'pca': [PCA(n_components=50), 'passthrough'],
        'regressor': [Ridge()],
        'regressor__alpha': [0.1, 1, 10, 50, 100]
    },
    # Ensemble Methods: RandomForestRegressor
    {
        'poly__degree': [2],
        'scaler': [StandardScaler(), RobustScaler()],
        'pca': [PCA(n_components=50), 'passthrough'],
        'regressor': [RandomForestRegressor(random_state=42)],
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20]
    },
    # Ensemble Methods: GradientBoostingRegressor
    {
        'poly__degree': [2],
        'scaler': [StandardScaler(), RobustScaler()],
        'pca': [PCA(n_components=50), 'passthrough'],
        'regressor': [GradientBoostingRegressor(random_state=42)],
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 5]
    }
]

# Run GridSearchCV with 5-fold cross-validation using negative MSE as the scoring metric
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display the best hyperparameters and corresponding cross-validation score
print("Best parameters:", grid_search.best_params_)
print("Best CV negative MSE:", grid_search.best_score_)

# Evaluate model performance on the validation set
y_val_pred = grid_search.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
print("Validation MSE:", val_mse)

# Train the final model using the full dataset if validation results look good
final_model = grid_search.best_estimator_
final_model.fit(X, y)

# Generate predictions for the test set
test_predictions = final_model.predict(x_test[feature_cols])

# Create a submission file
submission = pd.DataFrame({
    'id': x_test['id'],
    'y': test_predictions
})
submission.to_csv('y_test.csv', index=False)

print("Submission file 'y_test.csv' created successfully!")
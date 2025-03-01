import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error


# Custom mini batch Gradient Descent function with momentum & L2 regularization
class OptimizedGradientDescentRegressor(BaseEstimator, RegressorMixin):
    # use 'self' to get the instance of the class and allows me to have attributes to exist across methods
    def __init__(self, lr=0.01, n_iter=100, batch_size=32, momentum=0.9, l2=0.01):
        # Learning rate
        self.lr = lr
        # Number of iterations for training
        self.n_iter = n_iter
        # Size of each mini-batch
        self.batch_size = batch_size
        # Momentum for acceleration
        self.momentum = momentum
        # L2 regularization term (to prevent overfitting)
        self.l2 = l2
        # Initialize weight vector, bias term, and momentum terms
        self.w = None
        self.b = None
        self.v_w = None
        self.v_b = None

    # Train the model using mini batch gradient descent with momentum and L2 regularization
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights, bias, momentum all to zero
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.v_w = np.zeros(n_features)
        self.v_b = 0.0

        # Iterate through the number of training iterations
        for i in range(self.n_iter):
            # Shuffle data in order to avoid patterns
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Process the dataset in mini batches
            for j in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                # Compute predictions
                y_pred = np.dot(X_batch, self.w) + self.b

                # Compute gradients dw and db
                dw = - (2 / self.batch_size) * np.dot(X_batch.T, (y_batch - y_pred)) + self.l2 * self.w
                db = - (2 / self.batch_size) * np.sum(y_batch - y_pred)

                # Update weights and bias with momentum
                self.v_w = self.momentum * self.v_w + self.lr * dw
                self.v_b = self.momentum * self.v_b + self.lr * db
                self.w -= self.v_w
                self.b -= self.v_b

        return self
    # Make the predictions using trained model
    def predict(self, X):
        return np.dot(X, self.w) + self.b


# Load Training and Test Data given in Kaggle
train_df = pd.read_csv('/Users/willarsenault/PycharmProjects/DS310P1/venv/train.csv')
test_df = pd.read_csv('/Users/willarsenault/PycharmProjects/DS310P1/venv/x_test.csv')

# Get/extract the features and target variable
# Define target variable
target_column = 'y'
# Define feature column names
feature_cols = [f"Col {i}" for i in range(1, 65)]
X = train_df[feature_cols].values
y = train_df[target_column].values
X_test = test_df[feature_cols].values
# Store test IDs for submission file
test_ids = test_df['id']

# Remove low variance features and apply PCA
# Remove features with very low variance
var_thresh = VarianceThreshold(threshold=0.0001)
# Apply the threshold to the training and validation/testing data
X_reduced = var_thresh.fit_transform(X)
X_test_reduced = var_thresh.transform(X_test)

# Reduce dimensionality to top 30 components
pca = PCA(n_components=30)
# Apply PCA to training and testing data
X_pca = pca.fit_transform(X_reduced)
X_test_pca = pca.transform(X_test_reduced)


# Normalize/standardize features to have zero mean and 1 variance
scaler = StandardScaler()
# Apply scaling to training and testing data
X_scaled = scaler.fit_transform(X_pca)
X_test_scaled = scaler.transform(X_test_pca)


# Hyperparameter tuning using GridSearch using different learning rates, amount
# of iterations, batch sizes, momentum, and l2 lambda's
param_grid = {
    "lr": [0.001, 0.002, 0.003],
    "n_iter": [120, 150, 180],
    "batch_size": [16, 32, 64],
    "momentum": [0.8, 0.9, 0.99],
    "l2": [0.001, 0.01, 0.1]
}

# Initialize model
model = OptimizedGradientDescentRegressor()

# Perform a grid search with 5-fold cross-validation in order to get best hyperparameters
# with summary and sped up computation
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
# Fit model with cross-validation
grid_search.fit(X_scaled, y)

# Store best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Output best results
print("\nBest Hyperparameters:", best_params)
print("Best CV MSE:", -grid_search.best_score_)


# Make predictions on test data
test_predictions = best_model.predict(X_test_scaled)

# Save predictions to CSV for submission
submission = pd.DataFrame({
    'id': test_ids,
    'y': test_predictions
})

submission.to_csv('y_test.csv', index=False)

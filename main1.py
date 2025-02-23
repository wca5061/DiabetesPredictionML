import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Compute Mean Squared Error
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Implement gradient descent for linear regression
def gradient_descent(X, y, lr=0.01, n_iter=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    losses = []

    for i in range(n_iter):
        y_pred = np.dot(X, w) + b
        dw = - (2 / n_samples) * np.dot(X.T, (y - y_pred))
        db = - (2 / n_samples) * np.sum(y - y_pred)

        w -= lr * dw
        b -= lr * db

        loss = compute_mse(y, y_pred)
        losses.append(loss)

        if i % 100 == 0:
            print(f"Iteration {i}, MSE: {loss}")

    return w, b, losses

# Predict function using learned parameters
def predict(X, w, b):
    return np.dot(X, w) + b

# --------------------
# Data Loading Section
# --------------------
train_df = pd.read_csv('/Users/willarsenault/PycharmProjects/DS310P1/venv/train.csv')
test_df = pd.read_csv('/Users/willarsenault/PycharmProjects/DS310P1/venv/x_test.csv')

feature_cols = [f"Col {i}" for i in range(1, 65)]
X = train_df[feature_cols].values
y = train_df['y'].values

X_test = test_df[feature_cols].values
test_ids = test_df['id']

# -------------------------
# Feature Scaling
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Train-Validation Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# -------------------------
# Run Gradient Descent on Training Data
# -------------------------
learning_rate = 0.01
iterations = 115

w, b, losses = gradient_descent(X_train, y_train, lr=learning_rate, n_iter=iterations)

# -------------------------
# Evaluate on Validation Set (Estimate Kaggle MSE)
# -------------------------
y_val_pred = predict(X_val, w, b)
mse_val = compute_mse(y_val, y_val_pred)
print("Estimated Kaggle Test MSE (Validation MSE):", mse_val)

# -------------------------
# Train on Full Training Data (Final Model)
# -------------------------
w_final, b_final, _ = gradient_descent(X_scaled, y, lr=learning_rate, n_iter=iterations)

# -------------------------
# Predict on Test Data
# -------------------------
test_predictions = predict(X_test_scaled, w_final, b_final)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'y': test_predictions
})
submission.to_csv('y_test.csv', index=False)
print("Submission file 'y_test.csv' has been created.")

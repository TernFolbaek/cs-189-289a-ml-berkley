import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from save_csv import results_to_csv

# Load data
data = np.load('../data/cifar10-data.npz')
X_train = data['training_data']
y_train = data['training_labels']
X_test = data['test_data']

# Set random seed for reproducibility
np.random.seed(42)

# Shuffle data
total_examples = X_train.shape[0]
indices = np.random.permutation(total_examples)
X_shuffled = X_train[indices]
y_shuffled = y_train[indices]

# Split data into training and validation sets
X_train_split = X_shuffled[45000:]
y_train_split = y_shuffled[45000:]
X_val_split = X_shuffled[:45000]
y_val_split = y_shuffled[:45000]

# Create a pipeline with MinMaxScaler, PCA, and SVC
pipeline = make_pipeline(
    MinMaxScaler(),
    PCA(n_components=500, random_state=42),  # Adjust n_components based on desired balance between speed and accuracy
    SVC(gamma='scale', random_state=42)  # Use 'scale' for gamma and set a random_state for reproducibility
)

# Fit the model
pipeline.fit(X_train_split, y_train_split)

# Make predictions on the validation set
y_pred = pipeline.predict(X_val_split)

# Print accuracy
print(accuracy_score(y_val_split, y_pred))

# Predict on the test set and save results
results_to_csv(pipeline.predict(X_test))

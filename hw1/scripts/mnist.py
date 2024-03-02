import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from save_csv import results_to_csv
data = np.load('../data/mnist-data.npz')

X_train = data['training_data']
y_train = data['training_labels']
X_test = data['test_data']

np.random.seed(42)

total_examples = X_train.shape[0]
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train.reshape(-1, 28*28)) 
X_test = scaler.transform(X_test.reshape(-1, 28*28))

indices = np.random.permutation(total_examples)
X_shuffled = X_train[indices]
y_shuffled = y_train[indices]

split_size = 10000

X_train_split = X_shuffled[50000:]
y_train_split = y_shuffled[50000:]

X_val_split = X_shuffled[:50000]
y_val_split = y_shuffled[:50000]




clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train_split, y_train_split)
y_pred = clf.predict(X_val_split)


print(accuracy_score(y_val_split, y_pred))

results_to_csv(clf.predict(X_test))
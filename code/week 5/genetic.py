import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.models import clone_model
import random

# Data (as pandas dataframes)
X = load_iris().data
y = load_iris().target

# Convert string labels to numeric values using Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert labels to one-hot encoding for categorical crossentropy
y_one_hot = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Standardize the features (optional but often recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a sequential model
model = Sequential([
    Dense(units=8, input_dim=4, activation='relu'),
    Dense(units=3, activation='softmax')
])

# TODO: Create a genetic algorithm to train the network

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
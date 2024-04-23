import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import layers

import tensorflow as tf
import matplotlib.pyplot as plt

SEED_VAL = 42

np.random.seed(SEED_VAL)
tf.random.set_seed(SEED_VAL)

# Data set Exploration
(X_train , y_train ),(X_test,y_test)= boston_housing.load_data()

print(X_train.shape)
print("\n")
print("Input features " ,X_train[0])
print("\n")
print ("Output traget " , y_train[0])

# Extract Features from the dataset

boston_features = {
    "Average Number of Rooms " : 5
}

X_train_id = X_train[:,boston_features["Average Number of Rooms "]]
print(X_train_id.shape)

X_test_id = X_test[:, boston_features["Average Number of Rooms "]]

# plot the features 
plt.figure(figsize=(15,5))
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Price [$K]")
plt.grid("on")
plt.scatter(X_train_id[:] , y_train , color = "green" , alpha =0.5)

model = Sequential()

# Define the model consisting of a single neuron
model.add(Dense(units = 1 , input_shape = (1,)))

# Display a summary of the model architecture
model.summary()

# Compile the model 

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.005), loss = "mse")

# Train the model 
hostory = model.fit(
    X_train_id,
    y_train,
    batch_size = 32,
    epochs = 101,
    validation_split = 0.3
)

# pot the training result 
def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim([0, 100])
    plt.ylim([0, 300])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

# Predict the median price of a home with [3, 4, 5, 6, 7] rooms.
x = np.array([3, 4, 5, 6, 7]).reshape(-1, 1)
y_pred = model.predict(x)
for idx in range(len(x)):
    print(f"Predicted price of a home with {x[idx]} rooms: ${int(y_pred[idx] * 10) / 10}K")

# Plot the model and data
np.linspace(3,9,10)

# Use the model to predict the dependent variable
y=model.predict(x)

def plot_data(x_data, y_data, x, y, title=None):
    
    plt.figure(figsize=(15,5))
    plt.scatter(x_data, y_data, label='Ground Truth', color='green', alpha=0.5)
    plt.plot(x, y, color='k', label='Model Predictions')
    plt.xlim([3,9])
    plt.ylim([0,60])
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Price [$K]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
plot_data(X_train_id, y_train, x, y, title='Training Dataset')
plot_data(X_test_id, y_test, x, y, title='Test Dataset')

plt.show()
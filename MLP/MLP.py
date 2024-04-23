import random 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import seaborn as sn

plt.rcParams["axes.titlesize"]=16
plt.rcParams["axes.labelsize"]=14
plt.rcParams["image.cmap"]="gray"

from tensorflow.keras.datasets import fashion_mnist

SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

(X_train_all,y_train_all),(X_test,y_test) = mnist.load_data()

X_valid = X_train_all[:10000]
X_train = X_train_all[10000:]

y_valid = y_train_all[:10000]
y_train = y_train_all[10000:]

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.axis(True)
    plt.imshow(X_train[i], cmap="gray")
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

X_train = X_train.reshape((X_train.shape[0],28*28))
X_train = X_train.astype("float32")/255

X_test = X_test.reshape((X_test.shape[0],28*28))
X_test = X_test.astype("float32")/255

X_valid = X_valid.reshape((X_valid.shape[0] , 28*28))
X_valid = X_valid.astype("float32")/255

# Load the Fashion MNIST dataset.
((X_train_fashion , y_train_fashion) , (_, _)) = fashion_mnist.load_data()

# The labels in the Fashion MNIST dataset are encoded as integers.
print(y_train_fashion[0:9])

y_train_onehot = to_categorical(y_train_fashion[0:9])
print(y_train_onehot)

# Convert integer labels to one-hot encoded vectors.
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test = to_categorical(y_test)

# Instantiate the model.
model = tf.keras.Sequential()

# Build the model.
model.add(Dense(128, activation = 'relu' , input_shape = (X_train.shape[1],)))
model.add(Dense(128 , activation = 'relu'))
model.add(Dense(10 , activation = "softmax"))

# Display the model summary.
model.summary()

model.compile(
    optimizer = "rmsprop" , 
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

training_results = model.fit(
    X_train,
    y_train,
    epochs = 21,
    batch_size = 64, 
    validation_data = (X_valid , y_valid)
)
def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]
        
    for idx, metric in enumerate(metrics):    
        ax.plot(metric, color=color[idx])
    
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, 20])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)   
    plt.show()
    plt.close()

# Retrieve training results.
train_loss = training_results.history["loss"]
train_acc  = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc  = training_results.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 0.5],
    metric_name=["Training Loss", "Validation Loss"],
    color=["g", "b"],
)

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.9, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
)
predictions = model.predict(X_test)
index = 0
print("Ground truth for test digit: ", y_test[index])
print("\n")
print("Predictions for each class:\n")
for i in range(10):
    print("digit:", i, " probability: ", predictions[index][i])

# Generate predictions for the test dataset.
predictions = model.predict(X_test)

# For each sample image in the test dataset, select the class label with the highest probability.
predicted_labels = np.argmax(predictions, axis=1)


# Convert one-hot encoded labels to integers.
y_test_integer_labels = tf.argmax(y_test, axis =1)

# Generate a confusion matrix for the test dataset.
cm = tf.math.confusion_matrix(labels = y_test_integer_labels, predictions = predicted_labels)

# Plot the confusion matrix as a heatmap.
plt.figure(figsize=[15, 8])
import seaborn as sn

sn.heatmap(cm , annot = True , fmt = "d" , annot_kws = {"size" : 15})
plt.xlabel("Predicted")
plt.ylabel("Truth")



plt.show()
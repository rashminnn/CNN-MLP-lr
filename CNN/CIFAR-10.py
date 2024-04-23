import os 
import random
import numpy as np
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D, Dropout, Flatten

from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator,FormatStrFormatter)
from dataclasses import dataclass
from tensorflow.keras import models
import seaborn as sn

SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

(X_train,y_train) , (X_test,y_test) = cifar10.load_data()

print(X_train.shape)
print(X_test.shape)

plt.figure(figsize=(18,8))

num_rows=4
num_cols=8

# plot each of the images in the batch and the associated ground truth labels.
for i in range(num_rows*num_cols):
    ax=plt.subplot(num_rows,num_cols,i+1)
    plt.imshow(X_train[i,:,:])
    plt.axis("off")

# Normalize images to the range[0,1]
X_train = X_train.astype("float32")/255
X_test = X_test.astype("float32")/255

# Change the labels from integer to categorical data.
print('Original (integer) label for the first training sample: ', y_train[0])

# convert labels to one-hot encoding

y_train=to_categorical(y_train)
y_test = to_categorical(y_test)

print('After conversion to categorical one-hot encoded labels: ', y_train[0])

@dataclass(frozen = True)
class DatasetConfig:
    NUM_CLASSES : int =10
    IMG_HEIGHT: int=32
    IMG_WIDTH: int = 32
    NUM_CHANNELS: int = 3

@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS:    int=31
    BATCH_SIZE : int = 256
    LEARNING_RATE: float = 0.001


def cnn_model_dropout(input_shape = (32,32,3)):
    model = Sequential()

    # Conv block1: 32 filters , MaxPool.
    model.add(Conv2D(filters=32, kernel_size =3, padding ='same',activation= 'relu' , input_shape=input_shape))
    model.add(Conv2D(filters=32,kernel_size=3,padding = 'same' ,activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # Conv block1: 64 filters , MaxPool.
    model.add(Conv2D(filters=64,kernel_size = 3, padding='same', activation = 'relu'))
    model.add(Conv2D(filters=64,kernel_size = 3,padding = 'same' , activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # Conv block1: 64 filters , MaxPool.
    model.add(Conv2D(filters=64,kernel_size = 3, padding = 'same' , activation ='relu'))
    model.add(Conv2D(filters=64,kernel_size= 3, padding = 'same' , activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    # Flatten the convolutional features
    model.add(Flatten())
    model.add(Dense(512 , activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10,activation = 'softmax'))

    return model

# create the model

model_dropout = cnn_model_dropout()
model_dropout.summary()
# compile the model_dropout
model_dropout.compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
)


history = model_dropout.fit(
    X_train,
    y_train,
    batch_size = TrainingConfig.BATCH_SIZE,
    epochs = TrainingConfig.EPOCHS,
    verbose=1,
    validation_split = 0.3
)

# pot the training result
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
    plt.xlim([0, TrainingConfig.EPOCHS - 1])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()

# Retrieve training results.
train_loss =history.history["loss"]
train_acc = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc = history.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 5.0],
    metric_name=["Training Loss", "Validation Loss"],
    color=["g", "b"],
)

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.0, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
    )

# Using the save() method, the model will be saved to the file system in the 'SavedModel' format.
model_dropout.save("model_dropout.h5")

reloaded_model_dropout = tf.keras.models.load_model('model_dropout.h5')

test_loss , test_acc = reloaded_model_dropout.evaluate(X_test,y_test)
print(f"Test accuracy: {test_acc*100:.3f}")

def evaluate_model(dataset,model):
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_rows = 3
    num_cols = 6

# Retrieve a number of images from the dataset.
    data_batch = dataset[0:num_rows*num_cols]
# Get predictions from model.
    predictions = model.predict(data_batch)

    plt.figure(figsize=(20, 8))
    num_matches = 0

    for idx in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.axis("off")
        plt.imshow(data_batch[idx])

        pred_idx = tf.argmax(predictions[idx]).numpy()
        truth_idx = np.nonzero(y_test[idx])

        title = str(class_names[truth_idx[0][0]]) + " : " + str(class_names[pred_idx])
        title_obj = plt.title(title, fontdict={"fontsize": 13})

        if pred_idx == truth_idx:
            num_matches += 1
            plt.setp(title_obj, color="g")
        else:
            plt.setp(title_obj, color="r")

        acc = num_matches / (idx + 1)
    print("Prediction accuracy: ", int(100 * acc) / 100)

    return

evaluate_model(X_test,reloaded_model_dropout)

# confusion matrix
# Generate predictions for the test dataset

predictions = reloaded_model_dropout.predict(X_test)

predicted_labels =  [np.argmax(i) for i in predictions]

# onehot to int
y_test_integer_labels = tf.argmax(y_test,axis = 1)

cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

# plot the confusion matrix as a heat map.
plt.figure(figsize=[12,6])

sn.heatmap(cm , annot = True , fmt = "d" , annot_kws = {"size": 12})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

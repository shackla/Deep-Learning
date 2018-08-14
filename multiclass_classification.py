from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# since we are using multiple catagorical labels (i.e. more than 2 outputs)...
# labels must encoded to tensors using one hot encoding
# exact same as the vectorize function (different dimensions)
# fills tensor with 0s, columns = catagory
# each row will be all zeros except for the ..
# column that correspnds to its correct label
# the labels can be encoded as integer array(like in binary clasif.)
# onlyy difference is the loss function that is used...
# sparse_categorical_crossentropy- same as one used in this example..
# just a different interface


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# since we have 46 possible outcomes we need to make sure that...
# our intermediate dense layers have more than 46 points
# if the layers are too small the information can bottleneck
# the final layer has 46 points to represent the outcomes and uses..
# a softmax activation function
# soft max functions create a probabilty distrubtion over the 46 outcomes
# all probablities of the outcomes sum to 1

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# compile options stay mostly the same
# categorical_crossentropy is used as the loss function
# measure the difference between 2 probability distributions
# in this case it measures the difference between the model output...
# and the actually distrubtion of the labels

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# for this network , overtraining begins around the 9th epochs
# more possible outcome = more epochs needed

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=9,
                    batch_size=512,
                    validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(loss) + 1)

plt.figure(1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

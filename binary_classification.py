from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# This example will classify imdb reviews as either postive or negative
# The words in the review are encoded as integers (1-10000)(most common words)
# Rare words are ignored
# labels are either 0- negetive or 1- postive

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# data preparation:
# lists of integers cannot be fed into  nueral network
# must be turned into tensors
# method shown below turns a sequence of integers e.g [3,5]...
# into 10000 diminsonal vector of 0s and 1s
# 1 corresonds to the 3 and 5 and every other entry would be 0
# hence our lists have been vectorized
# each row of our vectorised input is one of the samples...
# each column represents of of the_words


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# the structure of our network consists of ...
# two intermediate layer with 16 hidden units...
# using the ReLU activation function:
# ReLU = max(x,0)
# and a final thrid layer that outputs the sentimant prediction
# the final layer uses a sigmoid activation function...
# which squished any number between 0-1

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(1, len(acc) + 1)

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

# this model is good example to over fitting...
# the training losses reduced with each epoch...
# however the validation losses seem to peak around the 4th epoch
# over training the model on the same data set can be counter productive
# the model over optimises itself to the training data set

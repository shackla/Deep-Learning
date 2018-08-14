from keras.datasets import mnist
# imports the mnist handwritten numbers data base from keras
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# loads both the pictures and labels to train the model (60000)
# and the the pictures and labels to test it (10000)
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
# layers are the building blocks of the network
# Dense -  densely/fully connected layers
# The 2nd (Last) layer is a 10-way softmax layer...
# it returns an array of 10 probabilities each corresponding a digit 0-9
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# loss - loss function = measure of performance on the training datasheets
# optimizer - the way the system adjusts through training (Backpropagation)
# metric - how the system knows if its doing a good job

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# categorically encodes the test_labels

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# before the data can be fed into the model it needs to be reformatted
# training data is stored in a (60000, 28,28) array
# 60000 images, 28 x 28 pixels, each pixel having a value between 0 -255
# 0 = black , 255 = white
# to use the data in the model the array must be reshped to (60000,28 *28)
# basically this invovles turning each 28*28 pixel image into a 784 * 1 line
# and the pixel values must be normalized between 0-1
# same must be done for the test datasheets

network.fit(train_images, train_labels, epochs=5, batch_size=128)
# network training begins
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
# evalulate the accuracy of the network

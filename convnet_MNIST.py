# simple convolutional nueral network(2 conv layers)
# CNN's differ from the densely connected as they look...
# for patterns in small windows (filters) and run them over (convolve)...
# the larger images
# the filters can detect patterns that are translationally invariant
# this means that the filters are able to detect the pattern that...
# they are set to detect in different parts of different samples
# this makes them particularly useful for image classification...
# the filters can recognise general patterns better as the ...
# densely connected network would have adapt all of its parameters for...
# each variation of the same class of image


from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train images reshaped as tensor 60000 images, 28*28 pixels, 1 colour channel
# pixel colour value normalised between 0-1
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# each 2d conv layer defines the size of the conv filter (3x3)..
# and the number of filters in each layer (32 for the first ,64 for the next 2)
# Max pooling functions essentially reduces the input shape by half
# this reduces the number of feature map coeffiecents in the layers
# removing this step would lead to the model having a huge number..
# of parameters in the final layer leading to massive overfitting
# flatten- flattens the tensor out into a vector
# the two dense layers work as with a normal classifier

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3),
                        activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64)

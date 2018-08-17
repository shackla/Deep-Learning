# this scripts shows the visual pattern that each filter responds to
# this can be achieved with gradient ascent in input space

from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# defining the model to use and the loss function

model = VGG16(weights='imagenet',
              include_top=False)
layer_name = 'block3_conv1'
filter_index = 0
layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])


grads = K.gradients(loss, model.input)[0]

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
step = 1.

# loop for stoichastic gradient descent

for i in range(40):
    loss_value, grads_value = iterate([input_img_data])

    input_img_data += grads_value * step

# deprocess_image -Normalises the tensor and centers around 0
# also ensures that the std = 0.1, clips to [0,1] and converts to an RGB array


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# builds a loss function that maxs the activation of the filters
# computes the gradient of the input img ,normalizes gradients
# img start as grey with noise then gradient ascent is run for 40 steps


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


# prints out all of the filter activations for the layer


layer_name = 'block1_conv1'
size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,
                vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()

# looking at each of the filters layer by layer...
# the first layer filters encode simple edges a colours
# next layer returns simple textures made from these colours and edges
# as you go deeper the filter encode more natural looking patterns...
# e.g scales feather furr etc...

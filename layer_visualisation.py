from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models

model = load_model('cats_and_dogs_small_2.h5')
model.summary()

base_dir = '/home/alex/Documents/MAI/Deep Learning/cats_dogs_small'

img_path = os.path.join(base_dir, 'test/dogs/dog.1916.jpg')

# loading image and converting to a tensor

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
plt.imshow(img_tensor[0])

# layer_outputs - extracts the outputs of the top 8 layer
# activation_model - creates a model that will return the outputs..
# given the model input

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# shows off what the 4th filter in the 1st layer is looking for..
# looks like it detects diagonal lines

first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

# this part of the code prints out all of the different filters for all layers

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            x = col * images_per_row + row
            channel_image = layer_activation[0, :, :, x]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size:(col + 1) * size,
                         row * size:(row + 1) * size] = channel_image

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()

# the first layer of activation acts as an edge detector
# as you go deeper the layers become less visual and more...
# about the class of the image
# its not detect visual aspects of image, its dealling with...
# more abstract class specfic aspects i.e ears, noses
# also as you do down the layers, less of them arre activated...
# meaning that encoded pattern they have is not being found in the input

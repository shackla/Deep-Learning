from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
from keras import backend as K
import matplotlib.pyplot as plt
import cv2

# importing the vgg16 - with the classifier this time

model = VGG16(weights='imagenet')

base_dir = '/home/alex/Documents/MAI/Deep Learning/cats_dogs_small/test/dogs'
img_path = os.path.join(base_dir, 'dog.1964.jpg')

# loading the test image and turn it into a usable array

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# use the model to predict the class of animal(92%-african elephant)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])


# african elephant position in prediction vector

african_elephant_output = model.output[:, np.argmax(preds[0])]
last_conv_layer = model.get_layer('block5_conv3')


grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# uses opencv to open images
# heat map is resized to same size as original
# converts the heat map to RGB
# heat map is applied to orginal image with intensity of 0.4
# saves image

img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('/home/alex/Documents/MAI/Deep Learning/dog_cam.jpg',
            superimposed_img)
plt.show()

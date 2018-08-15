# data augmentation can be used if there are too few sample availble..
# to train the model properly, augmenting the images allows more data to be..
# generated and used from the training, as long as the augmented data is not..
# too distroted, this helps generalise the model in the absence of large...
# amounts of training data


from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os

base_dir = '/home/alex/Documents/MAI/Deep Learning/cats_dogs_small'
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')

# in keras we can setup a random image transformer by using the ..
# ImageDataGenerator function
# rotation_range - (0-180) how much the image can be rotated
# height/width_shift_range = how much the picture can be moved..
# around (fraction of total height/width)
# shear_range - applies shear tranformations
# zoom_range - randomly zooms in on the image
# horizontal_flip - randomly flips half of the images horizontally
# fill_mode - method used to fill the empty pixels can can occur..
# after rotatiosn and height/width shifts (nearest means that the..
# nearest pixels to the empty area are copied to the empty ones)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]

# one image is chosen to augment

img_path = fnames[3]

# reads and resizes the image

img = image.load_img(img_path, target_size=(150, 150))

# converts to Numpy array and reshapes to (1,150,150,3)
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

# generates a batch of randomly augmented images

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

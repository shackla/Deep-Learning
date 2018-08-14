# this neural network will predict the price of houses in boston in the 70s
# there is relatively few data points in the training and testing datasets
# 406 training points, 102 test points
# there are 13 numerical evalution points for each training/test point...
# crime rate, distance from motorway, average num of rooms etc...
# all of them have different scales(0-100,0-1,1-12, etc... )


from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


(train_data, train_targets), (test_data,
                              test_targets) = boston_housing.load_data()

# data needs to be normalized before it is fed into the models
# the model might be able to handle the different ranges but it would...
# make learning quite a bit more challenging
# data normalised around the mean of TRAINING data never test data

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# new loss function mse -mean squared error
# calculates the losses on mean squared error between output and predictions
# new metric - MAE- mean absolute error
# calculates the absolute error between output and predictions
# MAE = 1 corresonds to $1000 difference between prediction and output
# final layer is 1 unit thick with no actvation
# this is common for regression problems that require a scaler linear output


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# due to the small number of data points k-fold validation should..
# be used instead of just spilting the sample set into a training and...
# validation set as the scores would change dramatically depending..
# on how the data is partitioned
# k-fold validation works by spilting the data in a number (K)...
# partitions and using K-1 partions to train the model and...
# validate using the other partition
# this process loops, changing the validation set each time
# the scores are then averaged to produce the final model


k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

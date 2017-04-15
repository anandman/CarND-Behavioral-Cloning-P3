import argparse
import json
import numpy as np
import pandas as pd
from os.path import expanduser, dirname
from sys import exit
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Conv2D, ELU, Dropout
from keras.utils import plot_model


# PARAMETERS FOR US TO SET (can also be set via command line flags)
# location of training data
DRIVING_LOG = expanduser("~/p3-sample-data/driving_log.csv")
# batch size to use in generator
BATCH_SIZE = 256
# total samples per epoch
EPOCH_SIZE = 16384
# number of epochs to run
EPOCHS = 5
# model to train with
MODEL = "comma"
# should we show the plots or not (e.g., turn off for non-interactive use)
SHOW_PLOTS = False
# offset of steering angle for left/right -> center
STEERING_OFFSET = 0.25


def read_driving_log(drivelog):
    """Reads in driving log into Pandas DataFrame"""

    # read in the driving log
    print("Reading in driving log...")
    # dl.center - path to center image
    # dl.left - path to left image
    # dl.right - path to right image
    # dl.steering - steering angle
    # dl.throttle - throttle amount
    # dl.brake - brake amount
    # dl.speed - speed
    dl = pd.read_csv(drivelog)

    data_path = dirname(drivelog) + "/"

    return dl, data_path


def data_generator(dl, data_path, batch_size=128):
    """Generator to read in, augment, and pre-process image data and yield in batches of batch_size"""

    # read in images & angles
    while 1:
        # shuffle order of log randomly
        rdl = dl.sample(frac=1).reset_index(drop=True)

        images = []
        angles = []
        for i in range(batch_size):
            # left = mpimg.imread(data_path + rdl.left[i].strip())
            center = mpimg.imread(data_path + rdl.center[i].strip())
            # right = mpimg.imread(data_path + rdl.right[i].strip())
            if rdl.flip[i]:
                center = np.fliplr(center)
            # images.append(left)
            images.append(center)
            # images.append(right)
            angles.append(rdl.steering[i])
        yield np.array(images), np.array(angles)


def simple_test_model():
    """Very simple model to test out flow"""

    # create model
    model = Sequential()

    # normalize pixels to between -1 <= x <= 1
    model.add(Lambda(lambda x: (x / 255.0) * 2 - 1, input_shape=(160,320,3)))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")

    return model


def comma_ai_model():
    """Steering model from comma.ai found at https://github.com/commaai/research/blob/master/train_steering_model.py"""

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
    model.add(Conv2D(16, (8, 8), padding="same", strides=(4, 4), activation="elu"))
    # model.add(ELU())
    model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2), activation="elu"))
    # model.add(ELU())
    model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def nvidia_model():
    """Steering model from NVIDIA paper at https://arxiv.org/pdf/1604.07316.pdf"""

    # had to take a guess at activation functions, optimizer, and loss function
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) * 2 - 1, input_shape=(160,320,3)))
    model.add(Conv2D(24, (5, 5), strides= (2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides= (2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides= (2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides= (1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides= (1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def train(dl, data_path, batch_size=BATCH_SIZE, epoch_size=EPOCH_SIZE, epochs=EPOCHS, modelname=MODEL):
    """Train "epoch_size" samples for "epoch" epochs with a batch size of "batch_size" on model "modelname"."""

    if modelname == 'simple':
        model = simple_test_model()
    elif modelname == 'comma':
        model = comma_ai_model()
    elif modelname == 'nvidia':
        model = nvidia_model()
    else:
        print('ERROR: model "{0}" is not implemented!'.format(modelname))
        exit(1)

    # train model
    print("Training {0} model...".format(modelname))
    trainer = data_generator(dl, data_path, batch_size=batch_size)
    validator = data_generator(dl, data_path, batch_size=batch_size)
    steps_per_epoch = epoch_size / batch_size
    validation_steps = 0.2 * epoch_size / batch_size
    hist = model.fit_generator(trainer, steps_per_epoch=steps_per_epoch, epochs=epochs,
                               validation_data=validator, validation_steps=validation_steps)

    # save model
    print("Saving {0} model...".format(modelname))
    # file names
    model_plot = "{0}.png".format(modelname)
    model_weights = "{0}.h5".format(modelname)
    model_keras = "{0}.json".format(modelname)
    # save graph of model to file
    plot_model(model, to_file=model_plot, show_shapes=True, show_layer_names=False)
    # save model weights
    model.save(model_weights)
    # save Keras model
    with open(model_keras, 'w') as outfile:
        json.dump(model.to_json(),outfile)

    return hist


def plot_training_history(hist, filename=''):
    """Plot histogram of model history"""

    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    if filename:
        plt.savefig(filename)
    if args.showplots:
        plt.show()
    plt.close()



def plot_histogram(arr, label="Targets", filename=''):
    """Plot histogram of 1D array"""

    bins, counts = np.unique(arr, return_counts=True)
    plt.bar(bins, counts, align='center')
    plt.ylabel(label + ' Count')
    plt.xlabel(label + ' Bins')
    if filename:
        plt.savefig(filename)
    if args.showplots:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=EPOCH_SIZE, help='How many frames per epoch.')
    parser.add_argument('--model', type=str, default=MODEL, choices=['simple', 'comma', 'nvidia'],
                        help='Which model to use: [simple, comma, nvidia]')
    parser.add_argument('--drivelog', type=str, default=DRIVING_LOG, help='CSV log of driving images & control data')
    parser.add_argument('--showplots', type=bool, default=SHOW_PLOTS, help='Show plots in X.')
    args = parser.parse_args()

    # read in the driving log and plot some statistics
    dl, data_path = read_driving_log(args.drivelog)
    plot_histogram(dl.steering, label="Initial Steering Angle", filename="initial_angles.png")

    # let's add a virtual flip for every line so that we can "balance" out the right/left steering
    # we will adjust the steering angle here...
    #   but not read in or actually flip the image...
    #   we'll just add a dl.flip column to the DataFrame so we can know to flip the image(s) later
    fdl = dl.copy()
    dl['flip'] = pd.Series(np.zeros(len(dl), dtype=bool), index=dl.index)
    fdl['flip'] = pd.Series(np.ones(len(fdl), dtype=bool), index=fdl.index)
    fdl.steering = fdl.steering.apply(lambda x: x * -1)
    dl = dl.append(fdl, ignore_index=True)

    # for dl_index, dl_row in dl.iterrows():
    #    print(dl_row.steering, dl_row.flip)

    # plot the steering angle histogram again to check it
    plot_histogram(dl.steering, label="Augmented Steering Angle", filename="augmented_angles.png")

    history = train(dl, data_path, batch_size=args.batch, epoch_size=args.epochsize, epochs=args.epochs, modelname=args.model)
    plot_training_history(history, filename="loss.png")

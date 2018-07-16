import numpy as np
import pathlib
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator  # Currently not used


def train_test_split(X, y, test_split_ratio, random_state=0):
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_split = int(test_split_ratio * X.shape[0])
    train_indices, test_indices = indices[n_split:], indices[:n_split]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def vertical_flip_random(X, y):
    width = X.shape[2]
    do_flip = np.random.randint(2, size=y.shape[0]).astype(np.bool)
    X[do_flip, ...] = X[do_flip, :, ::-1, :]
    y[do_flip, 0] = width - 1 - y[do_flip, 0]


def height_width_shift_random(X, y, width_range=2, height_range=2):
    batch_size, h, w = X.shape[:3]
    width_shift = np.random.randint(2 * width_range + 1, size=batch_size)
    height_shift = np.random.randint(2 * height_range + 1, size=batch_size)
    X_padded = np.pad(X, ((0, 0), (height_range, height_range), (width_range, width_range), (0, 0)), 'edge')
    for i in range(batch_size):
        X[i, ...] = X_padded[i, height_shift[i]: height_shift[i] + h, width_shift[i]: width_shift[i] + w, :]
    y -= np.array([width_shift - width_range, height_shift - height_range]).T


class DataGenerator:
    def __init__(self, config):
        self.config = config

        # Load data from files
        result_iuvr = np.loadtxt('./data/results_uvr.txt', dtype=np.str, delimiter=',')
        data_path = pathlib.Path('./data/images/')
        input_all = np.array([np.load(i) for i in data_path.glob('*.npy')])
        y_all = np.array([result_iuvr[result_iuvr[:, 0] == i.stem, 1:3].astype(np.float)[0] for i in
                          data_path.glob('*.npy')])

        # Split and rescale
        self.input, self.input_test, self.y, self.y_test = train_test_split(input_all, y_all, config.test_split_ratio)
        self.input, self.input_test = self.input.astype(np.float) / 255, self.input_test.astype(np.float) / 255

        # The following two lines of code would have been a better way to augment if it changed y also.
        # datagen = ImageDataGenerator(rescale=1./255, width_shift_range=4, height_shift_range=4, vertical_flip=True)
        # self.batch_generator = datagen.flow(self.input, self.y, batch_size=self.config.batch_size)

    def next_batch(self):
        idx = np.random.choice(self.y.shape[0], self.config.batch_size)
        X, y = self.input[idx], self.y[idx]
        vertical_flip_random(X, y)
        height_width_shift_random(X, y)
        return X, y

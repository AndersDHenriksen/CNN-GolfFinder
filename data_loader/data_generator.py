import numpy as np
import pathlib
import pandas as pd
import sklearn.model_selection


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        # self.input = np.ones((500, 784))
        # self.y = np.ones((500, 10))

        result_df = pd.read_csv('./data/results_uvr.txt', sep=',', names=('trigger_id', 'u', 'v', 'r'))
        result_df.set_index('trigger_id', inplace=True)

        data_path = pathlib.Path('./data/images/')
        input_all = np.array([np.load(i) for i in data_path.glob('*.npy')])
        y_all = np.array([np.array(result_df.loc[i.stem][['u', 'v']]) for i in data_path.glob('*.npy')])
        self.input, self.input_test, self.y, self.y_test = \
            sklearn.model_selection.train_test_split(input_all, y_all, test_size=config.test_split_ratio, random_state=0)

    def next_batch(self, batch_size):
        idx = np.random.choice(self.y.shape[0], batch_size)
        yield self.input[idx], self.y[idx]

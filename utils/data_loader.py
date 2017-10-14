import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

from utils.globals import DATA_SET_PATH, RGB_CHANNELS, IMAGE_DIM_X, IMAGE_DIM_Y


def unpickle(file):
    with open(file, 'rb') as fo:
        if (sys.version_info > (3, 0)):
            import pickle
            dict = pickle.load(fo, encoding='latin1')
        else:
            import cPickle
            dict = cPickle.load(fo)
    return dict


class DataSet(object):
    def __init__(self):
        self.meta = None
        self.data_batch = None
        self.test_batch = None

    def load_data(self, data_set_path):
        self.meta = unpickle(os.path.join(data_set_path, 'batches.meta'))
        self.data_batch = [self._get_path(data_set_path, x) for x in range(1, 6)]
        self.test_batch = unpickle(os.path.join(data_set_path, 'test_batch'))

    @staticmethod
    def _get_path(data_set_path, x):
        path = os.path.join(data_set_path, 'data_batch_' + str(x))
        return unpickle(path)

    def get_training_pictures(self):
        transformed_batch_list = [self._transform_batch_format(batch['data']) for batch in self.data_batch]
        return list(itertools.chain.from_iterable(transformed_batch_list))

    def get_training_labels(self):
        batch_labels = list(itertools.chain(*[batch['labels'] for batch in self.data_batch]))
        return [self.meta['label_names'][x] for x in batch_labels]

    def get_test_picutres(self):
        return self._transform_batch_format(self.test_batch['data'])

    def get_test_labels(self):
        return [self.meta['label_names'][x] for x in self.test_batch['labels']]


    @staticmethod
    def _transform_batch_format(x):
        return x.reshape(len(x), RGB_CHANNELS, IMAGE_DIM_X, IMAGE_DIM_Y).transpose(0, 2, 3, 1)


def get_data_set():
    data_set = DataSet()
    data_set.load_data(DATA_SET_PATH)
    return data_set


def visualize(x, y, grid_size=(4, 5)):
    fig, ax = plt.subplots(*grid_size, figsize=(12, 6))

    def plot_picture(dim_x, dim_y):
        chosen_image = np.random.choice(range(len(x)))
        ax[dim_x][dim_y].set_axis_off()
        ax[dim_x][dim_y].text(0, 36, y[chosen_image])
        ax[dim_x][dim_y].imshow(x[chosen_image])

    for i, j in itertools.product(range(grid_size[0]), range(grid_size[1])):
        plot_picture(i, j)

    plt.show()


if __name__ == "__main__":
    data_set = get_data_set()
    X = data_set.get_training_pictures()
    Y = data_set.get_training_labels()
    visualize(X, Y, (4, 10))

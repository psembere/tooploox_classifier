import os

from utils.globals import DATA_SET_NAME, DATA_SET_PATH


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


class DataSet(object):
    def __init__(self):
        self.meta = None
        self.data_batch = []
        self.test_batch = None

    def load_data(self, data_set_path):
        self.meta = unpickle(os.path.join(data_set_path, 'batches.meta'))
        for x in range(1,6):
            self.data_batch.append(unpickle(os.path.join(data_set_path, 'data_batch_' + str(x))))
        self.test_batch = unpickle(os.path.join(data_set_path, 'test_batch'))


if __name__ == "__main__":
    data_set = DataSet()
    data_set.load_data(DATA_SET_PATH)
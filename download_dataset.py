import os

from classifiers.globals import SERIALIZED_DATA_PATH, EXTRACTED_DATA_SET_DIR_NAME
from classifiers.inception_v3.download_inception import prepare_inception, INCEPTION_RESOURCES_PATH
from download_utils import download_data_set, extract_tar

TAR_DATA_SET_FILENAME = "cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/" + TAR_DATA_SET_FILENAME


def prepare_data():
    if not os.path.exists(EXTRACTED_DATA_SET_DIR_NAME):
        print("cifar10 data set downloading")
        download_data_set(URL, TAR_DATA_SET_FILENAME)
        extract_tar(TAR_DATA_SET_FILENAME)
        os.remove(TAR_DATA_SET_FILENAME)
    else:
        print("cifar10 data set loaded")

    if not os.path.exists(SERIALIZED_DATA_PATH):
        os.makedirs(SERIALIZED_DATA_PATH)

    if not os.path.exists(INCEPTION_RESOURCES_PATH):
        prepare_inception()
    else:
        print("inception prepared")


if __name__ == "__main__":
    prepare_data()

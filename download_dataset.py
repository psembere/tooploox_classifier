import os

from classifiers.globals import SERIALIZED_DATA_PATH, EXTRACTED_DATA_SET_DIR_NAME
from classifiers.inception_v3.download_utils import download_data_set, extract

TAR_DATA_SET_FILENAME = "cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/" + TAR_DATA_SET_FILENAME

if __name__ == "__main__":
    if not os.path.exists(EXTRACTED_DATA_SET_DIR_NAME):
        print ("data set downloading")
        download_data_set(URL, TAR_DATA_SET_FILENAME)
        extract(TAR_DATA_SET_FILENAME)
        os.remove(TAR_DATA_SET_FILENAME)
    else:
        print("data set loaded")

    if not os.path.exists(SERIALIZED_DATA_PATH):
        os.makedirs(SERIALIZED_DATA_PATH)

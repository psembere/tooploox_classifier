import os
import tarfile
import sys

from classifiers.globals import SERIALIZED_DATA_PATH

TAR_DATA_SET_FILENAME = "cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/" + TAR_DATA_SET_FILENAME


def download_data_set():
    if sys.version_info[0] < 3:
        import urllib
        urllib.URLopener().retrieve(URL, TAR_DATA_SET_FILENAME)
    else:
        import urllib.request
        urllib.request.urlretrieve(URL, TAR_DATA_SET_FILENAME)


def extract():
    tar = tarfile.open(TAR_DATA_SET_FILENAME, "r:gz")
    tar.extractall()
    tar.close()


if __name__ == "__main__":
    download_data_set()
    extract()
    os.remove(TAR_DATA_SET_FILENAME)

    if not os.path.exists(SERIALIZED_DATA_PATH):
        os.makedirs(SERIALIZED_DATA_PATH)

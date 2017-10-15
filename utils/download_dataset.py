import os
import tarfile
import urllib

from utils.globals import UTILS_DIR_PATH

DATA_SET_FILENAME_TAR = "cifar-10-python.tar.gz"
DATA_SET_FILENAME_PATH = os.path.join(UTILS_DIR_PATH, "..", DATA_SET_FILENAME_TAR)

URL = "https://www.cs.toronto.edu/~kriz/" + DATA_SET_FILENAME_TAR


def download_data_set():
    dataset_zipped = urllib.URLopener()
    dataset_zipped.retrieve(URL, DATA_SET_FILENAME_PATH)


def extract():
    tar = tarfile.open(DATA_SET_FILENAME_PATH, "r:gz")
    tar.extractall()
    tar.close()


if __name__ == "__main__":
    download_data_set()
    extract()
    os.remove(DATA_SET_FILENAME_PATH)

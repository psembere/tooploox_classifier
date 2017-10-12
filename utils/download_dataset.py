import tarfile
import urllib

import os

DATA_SET_FILENAME_TAR = "cifar-10-python.tar.gz"
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_SET_FILENAME_PATH = os.path.join(SCRIPT_DIR, "..", DATA_SET_FILENAME_TAR)

URL = "https://www.cs.toronto.edu/~kriz/" + DATA_SET_FILENAME_TAR


def download_dataset():
    dataset_zipped = urllib.URLopener()
    dataset_zipped.retrieve(URL, DATA_SET_FILENAME_PATH)


def extract():
    tar = tarfile.open(DATA_SET_FILENAME_PATH, "r:gz")
    tar.extractall()
    tar.close()


if __name__ == "__main__":
    download_dataset()
    extract()
    os.remove(DATA_SET_FILENAME_PATH)

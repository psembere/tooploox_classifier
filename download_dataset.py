import os
import tarfile
import urllib

TAR_DATA_SET_FILENAME = "cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/" + TAR_DATA_SET_FILENAME


def download_data_set():
    dataset_zipped = urllib.URLopener()
    dataset_zipped.retrieve(URL, TAR_DATA_SET_FILENAME)


def extract():
    tar = tarfile.open(TAR_DATA_SET_FILENAME, "r:gz")
    tar.extractall()
    tar.close()


if __name__ == "__main__":
    download_data_set()
    extract()
    os.remove(TAR_DATA_SET_FILENAME)

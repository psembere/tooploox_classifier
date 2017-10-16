import sys
import tarfile


def download_data_set(url, tar_file):
    if sys.version_info[0] < 3:
        import urllib
        urllib.URLopener().retrieve(url, tar_file)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, tar_file)


def extract_tar(tar_file, path="."):
    tar = tarfile.open(tar_file, "r:gz")
    tar.extractall(path)
    tar.close()
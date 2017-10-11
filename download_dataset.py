import tarfile
import urllib

FILENAME ="cifar-10-python.tar.gz"
URL = "https://www.cs.toronto.edu/~kriz/" + FILENAME

dataset_zipped = urllib.URLopener()
dataset_zipped.retrieve(URL, FILENAME)

tar = tarfile.open(FILENAME, "r:gz")
tar.extractall()
tar.close()

#"test"
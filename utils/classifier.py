CIFAR_FILE = "cifar-10-batches-py"


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


if __name__ == "__main__":
    meta = unpickle("../" + CIFAR_FILE + '/batches.meta')
    a=3

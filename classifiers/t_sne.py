from matplotlib import pyplot as plt
from time import time


def visualize_features_pypl_tsne_fast(features, number_of_features=1000):
    import numpy as np
    from tsne import bh_sne

    # load up data
    x_data = features.train_features
    y_data = features.train_labels

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # For speed of computation, only run on a subset
    x_data = x_data[:number_of_features]
    y_data = y_data[:number_of_features]

    # perform t-SNE embedding
    t0 = time()
    vis_data = bh_sne(x_data)
    t1 = time()

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    plt.clim(-0.5, 9.5)
    plt.show()


def visualize_features_sklearn_tsne(features, number_of_features=1000):
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_data = features.train_features[:number_of_features]
    y_data = features.train_labels[:number_of_features]

    t0 = time()
    Y = tsne.fit_transform(x_data)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    vis_x = Y[:, 0]
    vis_y = Y[:, 1]

    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    plt.show()

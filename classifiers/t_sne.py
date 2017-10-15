from matplotlib import pyplot as plt


def visualize_features_sklearn_tsne(features, number_of_features=1000):
    from time import time
    from sklearn import manifold

    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(features.train_features[0:1000])
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

    plt.scatter(Y[:, 0], Y[:, 1], c=features.train_labels[0:1000], cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    plt.axis('tight')
    plt.show()


def visualize_features_tsne(features, number_of_features=1000):
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
    vis_data = bh_sne(x_data)

    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()

from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np

from classifiers.inception_v3.download_inception import INCEPTION_PATH

model = INCEPTION_PATH + '/resources/classify_image_graph_def.pb'


def create_graph():
    """"Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    print 'Loading graph...'
    with tf.Session() as sess:
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    return sess.graph


def pool3_features(sess, X_input):
    """
    Call create_graph() before calling this
    """
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    pool3_features_tf = sess.run(pool3, {'DecodeJpeg:0': X_input[i, :]})
    return np.squeeze(pool3_features_tf)


def batch_pool3_features(sess, X_input):
    """
    Currently tensorflow can't extract pool3 in batch so this is slow:
    https://github.com/tensorflow/tensorflow/issues/1021
    """
    n_train = X_input.shape[0]
    print 'Extracting features for %i rows' % n_train
    create_graph()
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    X_pool3 = []
    for i in range(n_train):
        if i % 10000 ==0:
            print 'Iteration %i' % i
        pool3_features = sess.run(pool3, {'DecodeJpeg:0': X_input[i, :]})
        X_pool3.append(np.squeeze(pool3_features))
    return np.array(X_pool3)

import svm_utils.lib_svm.liblinearutil as liblinearutil


def linear_classifier(feature_set):
    prob = liblinearutil.problem(feature_set.train_labels, feature_set.train_features)
    param = liblinearutil.parameter('-c 4 -B 1')
    m = liblinearutil.train(prob, param)
    p_label, p_acc, p_val = liblinearutil.predict(feature_set.test_labels, feature_set.test_features, m)


class FeaturesDataSet(object):
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

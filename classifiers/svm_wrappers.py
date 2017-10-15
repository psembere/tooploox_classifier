import os

import svm_utils.lib_svm.liblinearutil as liblinearutil
import svm_utils.lib_svm.svmutil as svmutil
from classifiers.globals import PROJECT_PATH

linear_svm_path = os.path.join(PROJECT_PATH, "linear_svm")
kernel_svm_path = os.path.join(PROJECT_PATH, "kernel_svm")


def linear_classifier_generate(feature_set, params="", save=True):
    prob = liblinearutil.problem(feature_set.train_labels, feature_set.train_features)
    param = liblinearutil.parameter(params)
    m = liblinearutil.train(prob, param)
    p_label, p_acc, p_val = liblinearutil.predict(feature_set.test_labels, feature_set.test_features, m)
    if save:
        liblinearutil.save_model(linear_svm_path, m)
        print("lin svm model saved")
    print(p_label, p_acc, p_val)


def linear_classifier_evaluate(feature_set):
    m = liblinearutil.load_model(linear_svm_path)
    p_label, p_acc, p_val = liblinearutil.predict(feature_set.test_labels, feature_set.test_features, m)
    print("acc ", p_acc[0], "mean_square ", p_acc[1], " correlation", p_acc[2])


def kernel_classifier_generate(feature_set, params="h=0", save=True):
    prob = svmutil.svm_problem(feature_set.train_labels, feature_set.train_features)
    param = svmutil.svm_parameter(params)
    m = svmutil.svm_train(prob, param)
    p_label, p_acc, p_val = svmutil.svm_predict(feature_set.test_labels, feature_set.test_features, m)
    if save:
        svmutil.svm_save_model(linear_svm_path, m)
        print("kernel svm model saved")

    svmutil.svm_save_model(os.path.join(PROJECT_PATH, "kernel_svm"), m)
    print(p_label, p_acc, p_val)


def kernel_classifier_evaluate(feature_set):
    m = svmutil.svm_load_model(linear_svm_path)
    p_label, p_acc, p_val = svmutil.svm_predict(feature_set.test_labels, feature_set.test_features, m)
    print("acc ", p_acc[0], "mean_square ", p_acc[1], " correlation", p_acc[2])


class FeaturesDataSet(object):
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None

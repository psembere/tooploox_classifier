import os

import svm_utils.lib_svm.liblinearutil as liblinearutil
import svm_utils.lib_svm.svmutil as svmutil
from classifiers.globals import SERIALIZED_DATA_PATH

linear_svm_path = os.path.join(SERIALIZED_DATA_PATH, "linear_svm")
kernel_svm_path = os.path.join(SERIALIZED_DATA_PATH, "kernel_svm")
import time

class FeaturesDataSet(object):
    def __init__(self):
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None


class ClassifierGenerator(object):
    def __init__(self):
        self.training_preparator = None
        self.parameter_generator = None
        self.trainer = None
        self.predictor = None
        self.saver = None
        self.loader = None
        self.save_path = None
        self.model_info = None

    def classifier_generator(self, feature_set, params="", save=True):
        timestr = time.strftime("%d-%H-%M-%S")
        print timestr

        prob = self.training_preparator(feature_set.train_labels, feature_set.train_features)
        param = self.parameter_generator(params)
        m = self.trainer(prob, param)
        p_label, p_acc, p_val = self.predictor(feature_set.test_labels, feature_set.test_features, m)
        if save:
            path_to_save = self.save_path + self._get_timestamp()
            self.saver(self.save_path, m)
            print(self.model_info, " saved to ", path_to_save)
        print("Generated model details:")
        print("acc ", p_acc[0], "mean_square ", p_acc[1], " correlation", p_acc[2])

    @staticmethod
    def _get_timestamp():
        return time.strftime("-%d-%H-%M-%S")

    def evaluate(self, feature_set, path=None):
        loading_path = self.save_path if path is None else path
        m = self.loader(loading_path)
        p_label, p_acc, p_val = self.predictor(feature_set.test_labels, feature_set.test_features, m)
        print("Loaded model details:")
        print("acc ", p_acc[0], "mean_square ", p_acc[1], " correlation", p_acc[2])


class LinearClassifierGenerator(ClassifierGenerator):
    def __init__(self):
        super(LinearClassifierGenerator, self).__init__()
        self.training_preparator = liblinearutil.problem
        self.parameter_generator = liblinearutil.parameter
        self.trainer = liblinearutil.train
        self.predictor = liblinearutil.predict
        self.saver = liblinearutil.save_model
        self.loader = liblinearutil.load_model
        self.save_path = linear_svm_path
        self.model_info = "lin svm model"


class KernelClassifierGenerator(ClassifierGenerator):
    def __init__(self):
        super(KernelClassifierGenerator, self).__init__()
        self.training_preparator = svmutil.svm_problem
        self.parameter_generator = svmutil.svm_parameter
        self.trainer = svmutil.svm_train
        self.predictor = svmutil.svm_predict
        self.saver = svmutil.svm_save_model
        self.loader = svmutil.svm_load_model
        self.save_path = kernel_svm_path
        self.model_info = "kernel svm model"

def test_liblinear():
    from lib_svm.liblinearutil import svm_read_problem, train, predict, problem, parameter
    y, x = svm_read_problem('../heart_scale')
    m = train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = predict(y[200:], x[200:], m)

    y, x = [1, -1], [[1, 0, 1], [-1, 0, -1]]
    prob = problem(y, x)
    param = parameter('-c 4 -B 1')
    m = train(prob, param)


def test_libsvm():
    from lib_svm.svmutil import svm_read_problem, svm_train, svm_predict
    y, x = svm_read_problem('../heart_scale')
    m = svm_train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)


if __name__ == "__main__":
    test_liblinear()
    test_libsvm()

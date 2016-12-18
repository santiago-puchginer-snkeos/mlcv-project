import sklearn.preprocessing as preprocessing
import sklearn.svm as svm

import mlcv.io as io


def train_linear_svm(X, y, C=1, model_name=None, liblinear=False):
    # Standardize the data before classification
    std_scaler = preprocessing.StandardScaler().fit(X)
    X_std = std_scaler.transform(X)

    # Instance of SVM classifier
    clf = svm.LinearSVC(C=C, max_iter=5000, tol=1e-4) if liblinear else svm.SVC(kernel='linear', C=C)

    if model_name is not None:
        # Try to load a previously trained model
        try:
            clf = io.load_object(model_name)
        except (IOError, EOFError):
            clf.fit(X_std, y)
            # Store the model with the provided name
            io.save_object(clf, model_name)
    else:
        clf.fit(X_std, y)

    return clf, std_scaler


def predict_svm(X, svm, std_scaler=None):
    # Standardize data
    if std_scaler is None:
        X_std = X
    else:
        X_std = std_scaler.transform(X)

    # Predict the labels
    return svm.predict(X_std)

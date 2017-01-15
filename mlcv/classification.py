import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import sklearn.svm as svm

import mlcv.input_output as io
import mlcv.kernels as kernels


def train_linear_svm(X, y, C=1, standardize=True, dim_reduction=23, save_scaler=False, save_pca=False,
                     model_name=None, liblinear=False):
    # PCA for dimensionality reduction if necessary
    pca = None
    if dim_reduction is not None and dim_reduction > 0:
        pca = decomposition.PCA(n_components=dim_reduction)
        pca.fit(X)
        X = pca.transform(X)

    # Standardize the data before classification if necessary
    std_scaler = None
    if standardize:
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(X)
        X_std = std_scaler.transform(X)
    else:
        X_std = X

    # Instance of SVM classifier
    clf = svm.LinearSVC(C=C, max_iter=5000, tol=1e-4) if liblinear else svm.SVC(kernel='linear', C=C, probability=True)

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

    if save_scaler:
        io.save_object(std_scaler, save_scaler)

    if save_pca:
        io.save_object(pca, save_pca)

    return clf, std_scaler, pca


def train_poly_svm(X, y, C=1, degree=3, gamma='auto', coef0=0.0, standardize=True, dim_reduction=None,
                   save_scaler=False, save_pca=False, model_name=None):
    # PCA for dimensionality reduction if necessary
    pca = None
    if dim_reduction is not None and dim_reduction > 0:
        pca = decomposition.PCA(n_components=dim_reduction)
        pca.fit(X)
        X = pca.transform(X)

    # Standardize the data before classification if necessary
    std_scaler = None
    if standardize:
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(X)
        X_std = std_scaler.transform(X)
    else:
        X_std = X

    # Instance of SVM classifier
    clf = svm.SVC(kernel='poly', C=C, degree=degree, gamma=gamma, coef0=coef0, probability=True)

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

    if save_scaler:
        io.save_object(std_scaler, save_scaler)

    if save_pca:
        io.save_object(pca, save_pca)

    return clf, std_scaler, pca


def train_rbf_svm(X, y, C=5, gamma=0.1, standardize=True, dim_reduction=23,
                  save_scaler=False, save_pca=False, model_name=None):
    # PCA for dimensionality reduction if necessary
    pca = None
    if dim_reduction is not None and dim_reduction > 0:
        pca = decomposition.PCA(n_components=dim_reduction)
        pca.fit(X)
        X = pca.transform(X)

    # Standardize the data before classification if necessary
    std_scaler = None
    if standardize:
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(X)
        X_std = std_scaler.transform(X)
    else:
        X_std = X

    clf = svm.SVC(kernel='rbf', C=C, gamma=gamma, probability=True)

    if model_name is not None:
        # Instance of SVM classifier
        # Try to load a previously trained model
        try:
            clf = io.load_object(model_name)
        except (IOError, EOFError):
            clf.fit(X_std, y)
            # Store the model with the provided name
            io.save_object(clf, model_name)
    else:
        clf.fit(X_std, y)

    if save_scaler:
        io.save_object(std_scaler, save_scaler)

    if save_pca:
        io.save_object(pca, save_pca)

    return clf, std_scaler, pca


def train_sigmoid_svm(X, y, C=1, gamma='auto', coef0=0.0, standardize=True, dim_reduction=None,
                      save_scaler=False, save_pca=False, model_name=None):
    # PCA for dimensionality reduction if necessary
    pca = None
    if dim_reduction is not None and dim_reduction > 0:
        pca = decomposition.PCA(n_components=dim_reduction)
        pca.fit(X)
        X = pca.transform(X)

    # Standardize the data before classification if necessary
    std_scaler = None
    if standardize:
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(X)
        X_std = std_scaler.transform(X)
    else:
        X_std = X

    clf = svm.SVC(kernel='sigmoid', C=C, gamma=gamma, coef0=coef0, probability=True)

    if model_name is not None:
        # Instance of SVM classifier
        # Try to load a previously trained model
        try:
            clf = io.load_object(model_name)
        except (IOError, EOFError):
            clf.fit(X_std, y)
            # Store the model with the provided name
            io.save_object(clf, model_name)
    else:
        clf.fit(X_std, y)

    if save_scaler:
        io.save_object(std_scaler, save_scaler)

    if save_pca:
        io.save_object(pca, save_pca)

    return clf, std_scaler, pca


def train_intersection_svm(X, y, C=1, standardize=True, dim_reduction=None,
                           save_scaler=False, save_pca=False, model_name=None):
    # PCA for dimensionality reduction if necessary
    pca = None
    if dim_reduction is not None and dim_reduction > 0:
        pca = decomposition.PCA(n_components=dim_reduction)
        pca.fit(X)
        X = pca.transform(X)

    # Standardize the data before classification if necessary
    std_scaler = None
    if standardize:
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(X)
        X_std = std_scaler.transform(X)
    else:
        X_std = X

    clf = svm.SVC(kernel=kernels.intersection_kernel, C=C, probability=True)

    if model_name is not None:
        # Instance of SVM classifier
        # Try to load a previously trained model
        try:
            clf = io.load_object(model_name)
        except (IOError, EOFError):
            clf.fit(X_std, y)
            # Store the model with the provided name
            io.save_object(clf, model_name)
    else:
        clf.fit(X_std, y)

    if save_scaler:
        io.save_object(std_scaler, save_scaler)

    if save_pca:
        io.save_object(pca, save_pca)

    return clf, std_scaler, pca


def train_pyramid_svm(X, y, C=1, standardize=True, dim_reduction=None,
                           save_scaler=False, save_pca=False, model_name=None):

    # Standardize the data before classification if necessary
    std_scaler = None
    if standardize:
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(X)
        X_std = std_scaler.transform(X)
    else:
        X_std = X

    clf = svm.SVC(kernel=kernels.pyramid_kernel, C=C, probability=True)

    if model_name is not None:
        # Instance of SVM classifier
        # Try to load a previously trained model
        try:
            clf = io.load_object(model_name)
        except (IOError, EOFError):
            clf.fit(X_std, y)
            # Store the model with the provided name
            io.save_object(clf, model_name)
    else:
        clf.fit(X_std, y)

    if save_scaler:
        io.save_object(std_scaler, save_scaler)

    return clf, std_scaler, None


def predict_svm(X, svm, std_scaler=None, pca=None, probability=True):
    # Apply PCA if available
    if pca is not None:
        X = pca.transform(X)

    # Standardize data
    if std_scaler is None:
        X_std = X
    else:
        X_std = std_scaler.transform(X)

    # Predict the labels
    if probability:
        return svm.predict_proba(X_std)
    else:
        return svm.predict(X_std)


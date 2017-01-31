import os
import sys
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
IGNORE_PATH = 'ignore'


def load_training_set(load_images=False):
    """
    Loads the images that belong to the training set and their corresponding labels.

    :return: A tuple with 2 lists: the filenames of the train images, and their corresponding labels
    :rtype: tuple
    """
    with open(os.path.join(DATASET_PATH, 'train_images_filenames.dat'), 'r') as f:
        train_images_filenames = pickle.load(f)

    with open(os.path.join(DATASET_PATH, 'train_labels.dat'), 'r') as f:
        train_labels = pickle.load(f)

    if load_images:
        images = [load_image(fname) for fname in train_images_filenames]
        return np.array(images), train_labels
    else:
        return train_images_filenames, train_labels


def load_dataset_from_directory(directory):
    """
    Loads a dataset from a directory. The directory is expected to have subfolders, whose name
    indicates the class the images it contains belong to.

    :return: A tuple with an array of images, and their corresponding labels
    :rtype: tuple
    """
    images = []
    labels = []

    for label in os.listdir(directory):
        files_in_subfolder = os.listdir(os.path.join(directory, label))
        images += [os.path.join(directory, label, filepath) for filepath in files_in_subfolder]
        labels += [label] * len(files_in_subfolder)

    images = [load_image(fname) for fname in images]
    return np.array(images, dtype=np.float64), labels


def load_test_set(load_images=False):
    """
    Loads the images that belong to the test set and their corresponding labels.

    :return: A tuple with 2 lists: the filenames of the test images, and their corresponding labels
    :rtype: tuple
    """
    with open(os.path.join(DATASET_PATH, 'test_images_filenames.dat'), 'r') as f:
        test_images_filenames = pickle.load(f)

    with open(os.path.join(DATASET_PATH, 'test_labels.dat'), 'r') as f:
        test_labels = pickle.load(f)

    if load_images:
        images = [load_image(fname) for fname in test_images_filenames]
        return np.array(images), test_labels
    else:
        return test_images_filenames, test_labels


def load_image(image):
    try:
        import cv2
        img = cv2.imread(image)
    except ImportError:
        from skimage import data
        img = data.imread(image)

    return img


def load_grayscale_image(image):
    try:
        import cv2
        ima = cv2.imread(image)
        ima = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    except ImportError:
        from skimage import data
        ima = data.imread(image, as_grey=True)

    return ima


def save_object(obj, model_name, ignore=False):
    """
    Saves an object to disk

    :param obj: The object to be saved
    :type obj: object
    :param model_name: Name of the object to be saved
    :type model_name: basestring
    :param ignore: Store the object in the ignore folder
    :type ignore: bool
    """
    folder = IGNORE_PATH if ignore else MODELS_PATH
    filepath = os.path.join(folder, '{}.pickle'.format(model_name))
    try:
        import joblib
        joblib.dump(obj, filepath, compress=True)
    except ImportError:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)


def load_object(model_name, ignore=False):
    """
    Loads an object from disk

    :param model_name: Name of the model to be loaded
    :type model_name: basestring
    :param ignore: Load the object from the ignore folder
    :type ignore: bool
    :return: The object associated with loaded model
    :rtype: object, list
    """
    folder = IGNORE_PATH if ignore else MODELS_PATH
    filepath = os.path.join(folder, '{}.pickle'.format(model_name))
    try:
        import joblib
        obj = joblib.load(filepath)
    except ImportError:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

    return obj


def log(message='', out='stdout'):
    if out == 'stderr':
        sys.stderr.write('{}\n'.format(message))
        sys.stderr.flush()
    else:
        sys.stdout.write('{}\n'.format(message))
        sys.stdout.flush()


import os
import sys

#import cv2

try:
    import cPickle as pickle
except ImportError:
    import pickle

DATASET_PATH = 'dataset'
MODELS_PATH = 'models'
IGNORE_PATH = 'ignore'


def load_training_set():
    """
    Loads the images that belong to the training set and their corresponding labels.

    :return: A tuple with 2 lists: the filenames of the train images, and their corresponding labels
    :rtype: tuple
    """
    with open(os.path.join(DATASET_PATH, 'train_images_filenames.dat'), 'r') as f:
        train_images_filenames = pickle.load(f)

    with open(os.path.join(DATASET_PATH, 'train_labels.dat'), 'r') as f:
        train_labels = pickle.load(f)

    return train_images_filenames, train_labels


def load_test_set():
    """
    Loads the images that belong to the test set and their corresponding labels.

    :return: A tuple with 2 lists: the filenames of the test images, and their corresponding labels
    :rtype: tuple
    """
    with open(os.path.join(DATASET_PATH, 'test_images_filenames.dat'), 'r') as f:
        train_images_filenames = pickle.load(f)

    with open(os.path.join(DATASET_PATH, 'test_labels.dat'), 'r') as f:
        train_labels = pickle.load(f)

    return train_images_filenames, train_labels


#==============================================================================
# def load_image(image):
#     return cv2.imread(image)
# 
# 
# def load_grayscale_image(image):
#     ima = cv2.imread(image)
#     return cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
# 
#==============================================================================

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


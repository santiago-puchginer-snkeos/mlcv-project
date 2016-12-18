import cv2
import joblib
import numpy as np

import mlcv.input_output as io


def sift(gray):
    sift_fe = cv2.SIFT(nfeatures=100)
    kpt, des = sift_fe.detectAndCompute(gray, None)
    return kpt, des


def seq_sift(list_images_filenames, list_images_labels, num_samples_class=-1, **kwargs):
    descriptors = []
    label_per_descriptor = []
    image_id_per_descriptor = []

    for i, (filename, label) in enumerate(zip(list_images_filenames, list_images_labels)):
        # Check if we have limited the number of samples per class (not -1), and if so, only allow num_samples_class
        n_samples_class = label_per_descriptor.count(label)
        if num_samples_class == -1 or n_samples_class < num_samples_class:
            grayscale = io.load_grayscale_image(filename)
            kpt, des = sift(grayscale, **kwargs)
            descriptors.append(des)
            label_per_descriptor.append(label)
            image_id_per_descriptor.append(i)

    # Transform the descriptors and the labels to numpy arrays
    descriptors_matrix = descriptors[0]
    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
    for i in range(1, len(descriptors)):
        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))

    return descriptors_matrix, labels_matrix, indices_matrix


def compute_sift(ind, filename, label, **kwargs):
    grayscale = io.load_grayscale_image(filename)
    _, des = sift(grayscale, **kwargs)
    return des, label, ind


def parallel_sift(list_images_filenames, list_images_labels, num_samples_class=-1, n_jobs=4, **kwargs):
    descriptors = []
    label_per_descriptor = []
    image_id_per_descriptor = []

    if num_samples_class > 0:
        iterable_images = []
        iterable_labels_images = []
        for l in np.unique(list_images_labels):
            selection = [list_images_filenames[i] for i in range(len(list_images_filenames)) if
                         list_images_labels[i] == l]
            iterable_images += selection[:num_samples_class]
            iterable_labels_images += [l] * num_samples_class
        list_images_filenames = iterable_images
        list_images_labels = iterable_labels_images

    res = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
        joblib.delayed(compute_sift)(i, filename, label, **kwargs) for i, (filename, label) in
        enumerate(zip(list_images_filenames, list_images_labels))
    )

    for des, label, ind in res:
        descriptors.append(des)
        label_per_descriptor.append(label)
        image_id_per_descriptor.append(ind)

    # Transform the descriptors and the labels to numpy arrays
    descriptors_matrix = descriptors[0]
    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
    for i in range(1, len(descriptors)):
        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))

    return descriptors_matrix, labels_matrix, indices_matrix


def surf(gray):
    SURF_detector = cv2.SURF(hessianThreshold=2000)
    keypoints = SURF_detector.detect(gray, None)
    SURF_detector = cv2.SURF(hessianThreshold=300, nOctaves=4, extended=1, upright=0)
    kpt, des = SURF_detector.compute(gray, keypoints=keypoints)

    if len(kpt)>60:
        kp = kpt[0:60]
        desc = des[0:60, :]
    else:
        kp=kpt
        desc=des

    return kp, desc
def seq_surf(list_images_filenames, list_images_labels, num_samples_class=-1, **kwargs):
    descriptors = []
    label_per_descriptor = []
    image_id_per_descriptor = []

    for i, (filename, label) in enumerate(zip(list_images_filenames, list_images_labels)):
        # Check if we have limited the number of samples per class (not -1), and if so, only allow num_samples_class
        n_samples_class = label_per_descriptor.count(label)
        if num_samples_class == -1 or n_samples_class < num_samples_class:
            grayscale = io.load_grayscale_image(filename)
            kpt, des = surf(grayscale, **kwargs)
            descriptors.append(des)
            label_per_descriptor.append(label)
            image_id_per_descriptor.append(i)

    # Transform the descriptors and the labels to numpy arrays
    descriptors_matrix = descriptors[0]
    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
    for i in range(1, len(descriptors)):
        if not descriptors[i] is None:
            descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
            labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
            indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))

    return descriptors_matrix, labels_matrix, indices_matrix


def compute_surf(ind, filename, label, **kwargs):
    grayscale = io.load_grayscale_image(filename)
    _, des = surf(grayscale, **kwargs)
    return des, label, ind


def parallel_surf(list_images_filenames, list_images_labels, num_samples_class=-1, n_jobs=4, **kwargs):
    descriptors = []
    label_per_descriptor = []
    image_id_per_descriptor = []

    if num_samples_class > 0:
        iterable_images = []
        iterable_labels_images = []
        for l in np.unique(list_images_labels):
            selection = [list_images_filenames[i] for i in range(len(list_images_filenames)) if
                         list_images_labels[i] == l]
            iterable_images += selection[:num_samples_class]
            iterable_labels_images += [l] * num_samples_class
        list_images_filenames = iterable_images
        list_images_labels = iterable_labels_images

    res = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
        joblib.delayed(compute_surf)(i, filename, label, **kwargs) for i, (filename, label) in
        enumerate(zip(list_images_filenames, list_images_labels))
    )

    for des, label, ind in res:
        descriptors.append(des)
        label_per_descriptor.append(label)
        image_id_per_descriptor.append(ind)

    # Transform the descriptors and the labels to numpy arrays
    descriptors_matrix = descriptors[0]
    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
    for i in range(1, len(descriptors)):
        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))

    return descriptors_matrix, labels_matrix, indices_matrix



def orb(gray, n_features=100, levels=8, edge_threshold=31, wtak=2):
    orb_fe = cv2.ORB(nfeatures=n_features, nlevels=levels, edgeThreshold=edge_threshold, WTA_K=wtak)
    kp, des = orb_fe.detectAndCompute(gray, None)
    return kp, des


def brisk(gray, n_features=100):
    sift_fe = cv2.SIFT(nfeatures=n_features)
    keypoints = sift_fe.detect(gray, None)
    brisk_fe = cv2.DescriptorExtractor_create('brisk')
    kp, des = brisk_fe.compute(gray, keypoints)
    return kp, des


def brief(gray, n_features=100):
    detector = cv2.SIFT(nfeatures=n_features)
    brief_fe = cv2.DescriptorExtractor_create("brief")
    kp = detector.detect(gray)
    kp, des = brief_fe.compute(gray, kp)
    return kp, des


def freak(gray, n_features=100, sigma=1.6, contrast_threshold=0.04, edge_threshold=10):
    surf_detector = cv2.SIFT(nfeatures=n_features, sigma=sigma, contrastThreshold=contrast_threshold,
                             edgeThreshold=edge_threshold)
    keypoints = surf_detector.detect(gray, None)
    freak_extractor = cv2.DescriptorExtractor_create('freak')
    keypoints, descriptors = freak_extractor.compute(gray, keypoints)
    return keypoints, descriptors

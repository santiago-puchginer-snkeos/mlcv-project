import cv2
import numpy as np

import mlcv.io as io


def sift(gray, n_features=100, sigma=1.6, contrast_threshold=0.04, edge_threshold=10):
    sift_fe = cv2.SIFT(nfeatures=n_features, sigma=sigma, contrastThreshold=contrast_threshold,
                       edgeThreshold=edge_threshold)
    kpt, des = sift_fe.detectAndCompute(gray, None)
    return kpt, des


def seq_sift(list_images_filenames, list_images_labels, n_features=100, num_samples_class=-1):
    descriptors = []
    label_per_descriptor = []
    image_id_per_descriptor = []

    for i, (filename, label) in enumerate(zip(list_images_filenames, list_images_labels)):
        # Check if we have limited the number of samples per class (not -1), and if so, only allow num_samples_class
        n_samples_class = label_per_descriptor.count(label)
        if num_samples_class == -1 or n_samples_class <= num_samples_class:
            grayscale = io.load_grayscale_image(filename)
            kpt, des = sift(grayscale, n_features=n_features)
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


def parallel_sift(list_images_filenames, list_images_labels, n_jobs=4):
    pass


def surf(gray, n_features=100, sigma=1.6, contrast_threshold=0.04, edge_threshold=10):
    sift_fe = cv2.SIFT(nfeatures=n_features, sigma=sigma, contrastThreshold=contrast_threshold,
                       edgeThreshold=edge_threshold)
    keypoints = sift_fe.detect(gray, None)
    surf_fe = cv2.SURF(hessianThreshold=300, nOctaves=4, extended=1, upright=1)
    kpt, des = surf_fe.compute(gray, keypoints=keypoints)

    return kpt, des


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

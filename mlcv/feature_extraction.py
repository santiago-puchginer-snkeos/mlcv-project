#import cv2
import joblib
import numpy as np
import sklearn.decomposition as decomposition
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import mlcv.input_output as io
import mlcv.settings as settings


#def sift(gray, n_features=100):
#    sift_fe = cv2.SIFT(nfeatures=n_features)
#    kpt, des = sift_fe.detectAndCompute(gray, None)
#    return kpt, des
#
#
#def seq_sift(list_images_filenames, list_images_labels, num_samples_class=30):
#    descriptors = []
#    label_per_descriptor = []
#    image_id_per_descriptor = []
#
#    for i, (filename, label) in enumerate(zip(list_images_filenames, list_images_labels)):
#        # Check if we have limited the number of samples per class (not -1), and if so, only allow num_samples_class
#        n_samples_class = label_per_descriptor.count(label)
#        if num_samples_class == -1 or n_samples_class < num_samples_class:
#            grayscale = io.load_grayscale_image(filename)
#            kpt, des = sift(grayscale)
#            descriptors.append(des)
#            label_per_descriptor.append(label)
#            image_id_per_descriptor.append(i)
#
#    # Transform the descriptors and the labels to numpy arrays
#    descriptors_matrix = descriptors[0]
#    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
#    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
#    for i in range(1, len(descriptors)):
#        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
#        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
#        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))
#
#    return descriptors_matrix, labels_matrix, indices_matrix
#
#
#def compute_sift(ind, filename, label):
#    grayscale = io.load_grayscale_image(filename)
#    kp, des = sift(grayscale)
#    return des, label, ind, kp
#
#
#def parallel_sift(list_images_filenames, list_images_labels, num_samples_class=30, n_jobs=4, **kwargs):
#    descriptors = []
#    label_per_descriptor = []
#    image_id_per_descriptor = []
#    keypoints = []
#
#    if num_samples_class > 0:
#        iterable_images = []
#        iterable_labels_images = []
#        for l in np.unique(list_images_labels):
#            selection = [list_images_filenames[i] for i in range(len(list_images_filenames)) if
#                         list_images_labels[i] == l]
#            iterable_images += selection[:num_samples_class]
#            iterable_labels_images += [l] * num_samples_class
#        list_images_filenames = iterable_images
#        list_images_labels = iterable_labels_images
#
#    res = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
#        joblib.delayed(compute_sift)(i, filename, label, **kwargs) for i, (filename, label) in
#        enumerate(zip(list_images_filenames, list_images_labels))
#    )
#
#    for des, label, ind, kp in res:
#        descriptors.append(des)
#        label_per_descriptor.append(label)
#        image_id_per_descriptor.append(ind)
#        keypoints.append(np.array(kp))
#
#    # Transform the descriptors and the labels to numpy arrays
#    descriptors_matrix = descriptors[0]
#    keypoints_matrix = keypoints[0]
#    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
#    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
#    for i in range(1, len(descriptors)):
#        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
#        keypoints_matrix = np.hstack((keypoints_matrix, keypoints[i]))
#        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
#        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))
#
#    return descriptors_matrix, labels_matrix, indices_matrix, keypoints_matrix
#
#
#def surf(gray):
#    surf_detector = cv2.SURF(hessianThreshold=2000)
#    keypoints = surf_detector.detect(gray, None)
#    surf_detector = cv2.SURF(hessianThreshold=300, nOctaves=4, extended=1, upright=0)
#    kpt, des = surf_detector.compute(gray, keypoints=keypoints)
#
#    if len(kpt) > 60:
#        kp = kpt[0:60]
#        desc = des[0:60, :]
#    else:
#        kp = kpt
#        desc = des
#
#    return kp, desc
#
#
#def seq_surf(list_images_filenames, list_images_labels, num_samples_class=-1):
#    descriptors = []
#    label_per_descriptor = []
#    image_id_per_descriptor = []
#
#    for i, (filename, label) in enumerate(zip(list_images_filenames, list_images_labels)):
#        # Check if we have limited the number of samples per class (not -1), and if so, only allow num_samples_class
#        n_samples_class = label_per_descriptor.count(label)
#        if num_samples_class == -1 or n_samples_class < num_samples_class:
#            grayscale = io.load_grayscale_image(filename)
#            kpt, des = surf(grayscale)
#            descriptors.append(des)
#            label_per_descriptor.append(label)
#            image_id_per_descriptor.append(i)
#
#    # Transform the descriptors and the labels to numpy arrays
#    descriptors_matrix = descriptors[0]
#    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
#    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
#    for i in range(1, len(descriptors)):
#        if not descriptors[i] is None:
#            descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
#            labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
#            indices_matrix = np.hstack(
#                (indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))
#
#    return descriptors_matrix, labels_matrix, indices_matrix
#
#
#def compute_surf(ind, filename, label):
#    grayscale = io.load_grayscale_image(filename)
#    _, des = surf(grayscale)
#    return des, label, ind
#
#
#def parallel_surf(list_images_filenames, list_images_labels, num_samples_class=30, n_jobs=4, **kwargs):
#    descriptors = []
#    label_per_descriptor = []
#    image_id_per_descriptor = []
#
#    if num_samples_class > 0:
#        iterable_images = []
#        iterable_labels_images = []
#        for l in np.unique(list_images_labels):
#            selection = [list_images_filenames[i] for i in range(len(list_images_filenames)) if
#                         list_images_labels[i] == l]
#            iterable_images += selection[:num_samples_class]
#            iterable_labels_images += [l] * num_samples_class
#        list_images_filenames = iterable_images
#        list_images_labels = iterable_labels_images
#
#    res = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
#        joblib.delayed(compute_surf)(i, filename, label, **kwargs) for i, (filename, label) in
#        enumerate(zip(list_images_filenames, list_images_labels))
#    )
#
#    for des, label, ind in res:
#        if des is not None:
#            descriptors.append(des)
#            label_per_descriptor.append(label)
#            image_id_per_descriptor.append(ind)
#
#    # Transform the descriptors and the labels to numpy arrays
#    descriptors_matrix = descriptors[0]
#    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
#    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
#    for i in range(1, len(descriptors)):
#        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
#        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
#        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))
#
#    return descriptors_matrix, labels_matrix, indices_matrix
#
#
#def dense(gray):
#    dense = cv2.FeatureDetector_create("Dense")
#    dense.setInt('initXyStep', settings.dense_sampling_density)
#    kp = dense.detect(gray)
#    sift_detector = cv2.SIFT()
#    kp, des = sift_detector.compute(gray, kp)
#    return kp, des
#
#
#def seq_dense(list_images_filenames, list_images_labels, num_samples_class=-1):
#    descriptors = []
#    keypoints = []
#    label_per_descriptor = []
#    image_id_per_descriptor = []
#
#    for i, (filename, label) in enumerate(zip(list_images_filenames, list_images_labels)):
#        # Check if we have limited the number of samples per class (not -1), and if so, only allow num_samples_class
#        n_samples_class = label_per_descriptor.count(label)
#        if num_samples_class == -1 or n_samples_class < num_samples_class:
#            grayscale = io.load_grayscale_image(filename)
#            kpt, des = dense(grayscale)
#            descriptors.append(des)
#            label_per_descriptor.append(label)
#            image_id_per_descriptor.append(i)
#            keypoints.append(np.array(kpt))
#
#    # Transform the descriptors and the labels to numpy arrays
#    descriptors_matrix = descriptors[0]
#    keypoints_matrix = keypoints[0]
#    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
#    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
#    for i in range(1, len(descriptors)):
#        if not descriptors[i] is None:
#            descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
#            keypoints_matrix = np.hstack((keypoints_matrix, keypoints[i]))
#            labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
#            indices_matrix = np.hstack(
#                (indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))
#
#    return descriptors_matrix, labels_matrix, indices_matrix, keypoints_matrix
#
#
#def compute_dense(ind, filename, label):
#    grayscale = io.load_grayscale_image(filename)
#    kp, des = dense(grayscale)
#    return des, label, ind, kp
#
#
#def parallel_dense(list_images_filenames, list_images_labels, num_samples_class=30, n_jobs=4, **kwargs):
#    descriptors = []
#    label_per_descriptor = []
#    image_id_per_descriptor = []
#    keypoints = []
#
#    if num_samples_class > 0:
#        iterable_images = []
#        iterable_labels_images = []
#        for l in np.unique(list_images_labels):
#            selection = [list_images_filenames[i] for i in range(len(list_images_filenames)) if
#                         list_images_labels[i] == l]
#            iterable_images += selection[:num_samples_class]
#            iterable_labels_images += [l] * num_samples_class
#        list_images_filenames = iterable_images
#        list_images_labels = iterable_labels_images
#
#    res = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
#        joblib.delayed(compute_dense)(i, filename, label, **kwargs) for i, (filename, label) in
#        enumerate(zip(list_images_filenames, list_images_labels))
#    )
#
#    for des, label, ind, kp in res:
#        if des is not None:
#            descriptors.append(des)
#            label_per_descriptor.append(label)
#            image_id_per_descriptor.append(ind)
#            keypoints.append(np.array(kp))
#
#    # Transform the descriptors and the labels to numpy arrays
#    descriptors_matrix = descriptors[0]
#    keypoints_matrix = keypoints[0]
#    labels_matrix = np.array([label_per_descriptor[0]] * descriptors[0].shape[0])
#    indices_matrix = np.array([image_id_per_descriptor[0]] * descriptors[0].shape[0])
#    for i in range(1, len(descriptors)):
#        descriptors_matrix = np.vstack((descriptors_matrix, descriptors[i]))
#        keypoints_matrix = np.hstack((keypoints_matrix, keypoints[i]))
#        labels_matrix = np.hstack((labels_matrix, np.array([label_per_descriptor[i]] * descriptors[i].shape[0])))
#        indices_matrix = np.hstack((indices_matrix, np.array([image_id_per_descriptor[i]] * descriptors[i].shape[0])))
#
#    return descriptors_matrix, labels_matrix, indices_matrix, keypoints_matrix
#
#
def pca(descriptors_matrix):
    # Applying PCA
    pca = decomposition.PCA(n_components=settings.pca_reduction)
    pca.fit(descriptors_matrix)
    descriptors_matrix = pca.transform(descriptors_matrix)
    descriptors_matrix = np.float32(descriptors_matrix)


    return pca, descriptors_matrix

def CNN_features(x, model):
    features = model.predict(x)
    features = np.reshape(features, (features.shape[1], features.shape[2], features.shape[3]))
    features = features.reshape(14,14,16,32).swapaxes(1,2).reshape(14*14,512)
    features = features.transpose()
    return features
    

    
def compute_CNN_features(ind, filename, label, model):

    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    features = features[0, :, :, :]
    descriptor = features.reshape(features.shape[0] * features.shape[1], features.shape[2])

    return descriptor, label, ind


def parallel_CNN_features(list_images_filenames, list_images_labels, model, num_samples_class=-1, n_jobs=settings.n_jobs):
    descriptors = []
    label_per_descriptor = []
    image_id_per_descriptor = []
    keypoints = []

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

        joblib.delayed(compute_CNN_features)(i, filename, label, model) for i, (filename, label) in
        enumerate(zip(list_images_filenames, list_images_labels))
    )

    for des, label, ind in res:
        if des is not None:
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


#def orb(gray, n_features=100, levels=8, edge_threshold=31, wtak=2):
#    orb_fe = cv2.ORB(nfeatures=n_features, nlevels=levels, edgeThreshold=edge_threshold, WTA_K=wtak)
#    kp, des = orb_fe.detectAndCompute(gray, None)
#    return kp, des
#
#
#def brisk(gray, n_features=100):
#    sift_fe = cv2.SIFT(nfeatures=n_features)
#    keypoints = sift_fe.detect(gray, None)
#    brisk_fe = cv2.DescriptorExtractor_create('brisk')
#    kp, des = brisk_fe.compute(gray, keypoints)
#    return kp, des
#
#
#def brief(gray, n_features=100):
#    detector = cv2.SIFT(nfeatures=n_features)
#    brief_fe = cv2.DescriptorExtractor_create("brief")
#    kp = detector.detect(gray)
#    kp, des = brief_fe.compute(gray, kp)
#    return kp, des
#
#
#def freak(gray, n_features=100, sigma=1.6, contrast_threshold=0.04, edge_threshold=10):
#    surf_detector = cv2.SIFT(nfeatures=n_features, sigma=sigma, contrastThreshold=contrast_threshold,
#                             edgeThreshold=edge_threshold)
#    keypoints = surf_detector.detect(gray, None)
#    freak_extractor = cv2.DescriptorExtractor_create('freak')
#    keypoints, descriptors = freak_extractor.compute(gray, keypoints)
#    return keypoints, descriptors

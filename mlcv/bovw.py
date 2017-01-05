import numpy as np
import sklearn.cluster as cluster

import mlcv.input_output as io


def create_codebook(X, k=512, codebook_name=None, k_means_init='random'):
    batch_size = 20 * k if X.shape[0] > 20 * k else X.shape[0] / 10
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=batch_size, compute_labels=False,
                                       reassignment_ratio=10 ** -4, init=k_means_init)

    if codebook_name is not None:
        # Try to load a previously trained codebook
        try:
            codebook = io.load_object(codebook_name)
        except (IOError, EOFError):
            codebook.fit(X)
            # Store the model with the provided name
            io.save_object(codebook, codebook_name)
    else:
        codebook.fit(X)

    return codebook


def visual_words(X, y, descriptors_indices, codebook, normalization=None, spatial_pyramid=False, keypoints=None):
    k = codebook.cluster_centers_.shape[0]
    prediction = codebook.predict(X)

    if spatial_pyramid==False:
        v_words = np.array([np.bincount(prediction[descriptors_indices == i], minlength=k) for i in
                   range(0, descriptors_indices.max() + 1)], dtype=np.float64)
    elif spatial_pyramid==True:
        v_words = build_pyramid(prediction, descriptors_indices, k, keypoints)

    # Normalization
    if normalization == 'l1':
        vis_words = v_words / np.linalg.norm(v_words, axis=1, keepdims=True)
    elif normalization == 'l2':
        vis_words = v_words / np.linalg.norm(v_words, axis=1, keepdims=True) ** 2
    else:
        vis_words = v_words

    labels = [y[descriptors_indices == i][0] for i in
              range(0, descriptors_indices.max() + 1)]
    return vis_words, np.array(labels)


def build_pyramid(prediction, descriptors_indices, k, keypoints):

    v_words = []

    #Build representation for each image
    for i in range(0, descriptors_indices.max() + 1):

        image_predictions = prediction[descriptors_indices == i]
        image_keypoints = keypoints[descriptors_indices == i]

        # Level 0 - 4x4 grid
        level0_1_4 = []
        level0_5_8 = []
        level0_9_12 = []
        level0_13_16 = []

        for ini_i in range(0,129,128):
            for ini_j in range(0, 129, 128):
                level0_1_4.append(np.bincount(image_predictions[(image_keypoints[:, 0] >= ini_i) &
                                                                (image_keypoints[:, 0] < ini_i + 64) &
                                                                (image_keypoints[:, 1] >= ini_j) &
                                                                (image_keypoints[:, 1] < ini_j + 64)], minlength=k))

                level0_5_8.append(np.bincount(image_predictions[(image_keypoints[:, 0] >= ini_i) &
                                                                (image_keypoints[:, 0] < ini_i + 64) &
                                                                (image_keypoints[:, 1] >= ini_j + 64) &
                                                                (image_keypoints[:, 1] < ini_j + 128)], minlength=k))

                level0_9_12.append(np.bincount(image_predictions[(image_keypoints[:, 0] >= ini_i + 64) &
                                                                 (image_keypoints[:, 0] < ini_i + 128) &
                                                                 (image_keypoints[:, 1] >= ini_j) &
                                                                 (image_keypoints[:, 1] < ini_j + 64)], minlength=k))

                level0_13_16.append(np.bincount(image_predictions[(image_keypoints[:, 0] >= ini_i + 64) &
                                                                  (image_keypoints[:, 0] < ini_i + 128) &
                                                                  (image_keypoints[:, 1] >= ini_j + 64) &
                                                                  (image_keypoints[:, 1] < ini_j + 128)], minlength=k))
        # Level 1- 2x2 grid
        level1_1 = level0_1_4[0] + level0_5_8[0] + level0_9_12[0] + level0_13_16[0]
        level1_2 = level0_1_4[1] + level0_5_8[1] + level0_9_12[1] + level0_13_16[1]
        level1_3 = level0_1_4[2] + level0_5_8[2] + level0_9_12[2] + level0_13_16[2]
        level1_4 = level0_1_4[3] + level0_5_8[3] + level0_9_12[3] + level0_13_16[3]

        # Level 2 - whole image
        level2 = level1_1 + level1_2 + level1_3 + level1_4

        representation = np.hstack((0.25*level2,0.25*level1_1,0.25*level1_2,0.25*level1_3,0.25*level1_4))
        for g in range(0,4):
            representation = np.hstack((representation, 0.5*level0_1_4[g]))
            representation = np.hstack((representation, 0.5*level0_5_8[g]))
            representation = np.hstack((representation, 0.5*level0_9_12[g]))
            representation = np.hstack((representation, 0.5*level0_13_16[g]))

        v_words.append(representation)

    return np.array(v_words, dtype=np.float64)
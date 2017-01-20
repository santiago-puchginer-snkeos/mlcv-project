from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot

import mlcv.feature_extraction as feature_extraction
import mlcv.settings as settings
import mlcv.bovw as bovw
import mlcv.input_output as io
import mlcv.classification as classification
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


def parallel_testing(test_image, test_label, svm, scaler, gmm, model, pca):
    D = feature_extraction.compute_CNN_features(test_image, model)
    D_pca = pca.transform(D)
    D_pca = np.float32(D_pca)
    labels = np.array([test_label] * D.shape[0])
    ind = np.array([0] * D.shape[0])
    fisher, _ = bovw.fisher_vectors(D_pca, labels, ind, gmm, normalization='l2')
    prediction_prob = classification.predict_svm(fisher, svm, std_scaler=scaler)
    #print prediction_prob
    #print prediction_prob.shape
    predicted_class = svm.classes_[np.argmax(prediction_prob)]
    return predicted_class == test_label, predicted_class, np.ravel(prediction_prob)


best_accuracy = 0
best_params = {}
cv_results = {}

""" SETTINGS """
settings.n_jobs = 1
settings.codebook_size = 32


# Read the training set
train_images_filenames, train_labels = io.load_training_set()
io.log('Loaded {} train images.'.format(len(train_images_filenames)))


print('Loading VGG model...')
# load VGG model
base_model = VGG16(weights='imagenet')

# visualize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

# crop the model up to a certain layer
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)

print('Obtaining features...')
try:
    D, L, I = io.load_object('train_CNN_descriptors', ignore=True), \
              io.load_object('train_CNN_labels', ignore=True), \
              io.load_object('train_CNN_indices', ignore=True)
except IOError:

    D, L, I, _ = feature_extraction.parallel_CNN_features(train_images_filenames, train_labels,
                                                   num_samples_class=-1,
                                                   model=model,
                                                   n_jobs=settings.n_jobs)
    io.save_object(D, 'train_CNN_descriptors', ignore=True)
    io.save_object(L, 'train_CNN_labels', ignore=True)
    io.save_object(I, 'train_CNN_indices', ignore=True)

# get the features from images

settings.pca_reduction = D.shape[1]/2
pca, D_pca = feature_extraction.pca(D)

k = settings.codebook_size
gmm = bovw.create_gmm(D_pca, 'gmm_{}_CNNfeature'.format(k))
fisher, labels = bovw.fisher_vectors(D_pca, L, I, gmm, normalization='l2', spatial_pyramid=False)
#std_scaler = StandardScaler().fit(fisher)
#vis_words = std_scaler.transform(fisher)

print('Training the SVM classifier...')
lin_svm, std_scaler, _ = classification.train_intersection_svm(fisher, train_labels, C=0.0268101613883,  dim_reduction = None)


# Read the test set
test_images_filenames, test_labels = io.load_test_set()
print('Loaded {} test images.'.format(len(test_images_filenames)))

# Feature extraction with sift, prediction with SVM and aggregation to obtain final class
print('Predicting test data...')
test_results = joblib.Parallel(n_jobs=settings.n_jobs, backend='threading')(
    joblib.delayed(parallel_testing)(test_image, test_label, lin_svm, std_scaler, gmm, model, pca) for
    test_image, test_label in
    zip(test_images_filenames, test_labels))

pred_results = [xx[0] for xx in test_results]
pred_class = [xx[1] for xx in test_results]
pred_prob = [xx[2] for xx in test_results]

num_correct = np.count_nonzero(pred_results)


# Compute accuracy
accuracy = num_correct * 100.0 / len(test_images_filenames)

# Show results and timing
print('\nACCURACY: {:.2f}'.format(accuracy))


#if K.image_dim_ordering() == 'th':
    # theano and thensorflow deal with tensor in different order
#    pass

#weights = base_model.get_layer('block1_conv1').get_weights()




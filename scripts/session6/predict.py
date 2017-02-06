from __future__ import print_function, division

import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.models import load_model

import mlcv.cnn as cnn
import mlcv.input_output as io
import mlcv.plotting as plotting

""" MODEL """
model_file = '/home/master/santi/weights/compactparams_reg-0.1_awgn-0_dropout-None.hdf5'

""" CONSTANTS """
train_data_dir = './dataset/MIT_split/train'
test_data_dir = './dataset/MIT_split/test'
img_width = 128
img_height = 128
test_samples = 807
batch_size = 150

""" TEST GENERATOR """
# Data generators
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             preprocessing_function=cnn.preprocess_input)

train_images, train_labels = io.load_dataset_from_directory(train_data_dir)
test_images, test_labels = io.load_dataset_from_directory(test_data_dir)
datagen.fit(train_images)
test_generator = datagen.flow_from_directory(test_data_dir,
                                             shuffle=False,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

""" MODEL """
model = load_model(model_file)


""" TEST """
print('\n--------------------------------')
print('EVALUATING PERFORMANCE ON TEST SET')
print('--------------------------------\n')
result = model.evaluate_generator(test_generator, val_samples=test_samples)
print('Loss: {:.2f} \t Accuracy: {:.2f} %'.format(result[0], result[1] * 100))

print('\n--------------------------------')
print('COMPUTING CONFUSION MATRIX')
print('--------------------------------\n')
probs = model.predict_generator(test_generator, val_samples=test_samples)

classes = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
index_classes = np.argmax(probs, axis=1)
predicted_class = []
for i in index_classes:
    predicted_class.append(classes[i])
predicted_class = np.array(predicted_class)
conf = confusion_matrix(test_labels, predicted_class, labels=classes)
plotting.plot_confusion_matrix(conf, classes=classes, normalize=True)

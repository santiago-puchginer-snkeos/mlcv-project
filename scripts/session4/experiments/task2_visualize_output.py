# import matplotlib

#matplotlib.use('Agg')
#import pylab as plt
import numpy as np
import os
from matplotlib import pyplot as plt
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg19 import preprocess_input
# from keras.models import Model
try:
    import cPickle as pickle
except ImportError:
    import pickle

server=0
if server==1:

    img_path = './dataset/MIT_split/test/coast/art1130.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    base_model = VGG16(weights=None)
    model = Model(input=base_model.input, output=base_model.get_layer('block3_conv1').output)
    features = model.predict(x)
    features = features[0, :, :, :]

    energy = features.max(axis=0)
    energy=energy.max(axis=0)
    descriptor = features[:,:, np.argmax(energy)]

    folder = 'ignore'
    filepath = os.path.join(folder, '{}.pickle'.format('task2_energy'))



    DATASET_PATH = 'dataset'
    MODELS_PATH = 'models'
    IGNORE_PATH = 'ignore'
    with open(filepath, 'wb') as f:
        pickle.dump(descriptor, f)
elif server==0:
   with open('./ignore/task2_energy.pickle', 'rb') as f:
    im = pickle.load(f)

    fig, ax = plt.subplots()
    cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
    im = ax.imshow(im, cmap='jet')
    fig.colorbar(im, cax=cax, orientation='horizontal')
    plt.show()

from __future__ import print_function

import cPickle

train_images_filenames = cPickle.load(open('../dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('../dataset/test_images_filenames.dat', 'r'))

print('ORIGINAL FILENAMES')
print(train_images_filenames)
print(test_images_filenames)

train_images_filenames = map(lambda x: x.replace('../../Databases', './dataset'), train_images_filenames)
test_images_filenames = map(lambda x: x.replace('../../Databases', './dataset'), test_images_filenames)

print('\nTRANSFORMED FILENAMES')
print(train_images_filenames)
print(test_images_filenames)

print('\nSTORING TRANSFORMED LISTS')
cPickle.dump(train_images_filenames, open('../dataset/train_images_filenames_map.dat', 'wb'))
cPickle.dump(test_images_filenames, open('../dataset/test_images_filenames_map.dat', 'wb'))

print('\nDone')

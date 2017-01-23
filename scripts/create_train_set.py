from __future__ import print_function, division

import os
import math
import numpy as np
import shutil

if __name__ == '__main__':

    """ CONSTANTS """
    INPUT_DIR = '../dataset/MIT_split/train'
    NUM_SAMPLES = 40
    OUTPUT_DIR = '../dataset/{}_dataset'.format(NUM_SAMPLES)

    """ COMPUTE PRIORS """
    labels = os.listdir(INPUT_DIR)
    priors = dict()
    total_images = 0
    for label in labels:
        num_images = len(os.listdir(os.path.join(INPUT_DIR, label)))
        total_images += num_images
        priors[label] = num_images

    assert NUM_SAMPLES < total_images

    """ CREATE NEW DATASET OUT OF PRIORS """
    for label, num_images in priors.iteritems():
        # Compute the number of images to be randomly copied to the new dataset
        proportion = num_images / total_images
        selected_images = int(math.ceil(NUM_SAMPLES * proportion))
        print('\n{}'.format(label.upper()))
        print('Proportion in the original set: {:.2f} %'.format(proportion*100))
        print('Selection in the new set: {} out of {}'.format(
            selected_images,
            NUM_SAMPLES
        ))
        # Get the random selection of images
        images = os.listdir(os.path.join(INPUT_DIR, label))
        rand_permutation = np.random.permutation(images)
        selection = rand_permutation[:selected_images]

        # Create the destination folder
        dst_folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(os.path.join(OUTPUT_DIR, label))
        print('Destination folder created at: {}'.format(dst_folder))

        # Copy the selected files to the destionation folder
        src_folder = os.path.join(INPUT_DIR, label)
        for f in selection:
            dst = os.path.join(dst_folder, f)
            src = os.path.join(src_folder, f)
            shutil.copyfile(src, dst)
        print('All images copied')

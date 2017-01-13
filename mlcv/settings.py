import math
#Global variables to be accessed from anywhere in the program
#Their value may be changed at the beginning of the main script

image_size = [256,256]
codebook_size = 512

dense_sampling_density = 6

pyramid_levels=[[2,2],[4,4]]


def get_keypoints_shape():
    return [math.ceil(float(image_size[0])/float(dense_sampling_density)),math.ceil(float(image_size[0])/float(dense_sampling_density))]
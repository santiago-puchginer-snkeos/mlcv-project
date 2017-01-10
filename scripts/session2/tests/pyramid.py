from __future__ import print_function, division

import numpy as np

import mlcv.bovw as bovw

import mlcv.input_output as io
import cv2
from matplotlib import pyplot as plt


''' Test pyramids, plot histograms of different levels'''


K = 512

train_images_filenames, train_labels = io.load_training_set()

D, L, I, Kp_pos = io.load_object('train_sift_descriptors', ignore=True), \
                  io.load_object('train_sift_labels', ignore=True), \
                  io.load_object('train_sift_indices', ignore=True), \
                  io.load_object('train_sift_keypoints', ignore=True)

codebook = bovw.create_codebook(D, k=K, codebook_name='default_codebook')


X=D
y=L
descriptors_indices = I
keypoints=Kp_pos

k = codebook.cluster_centers_.shape[0]
prediction = codebook.predict(X)

v_words = []



image_predictions = prediction[descriptors_indices == 0]
image_keypoints = keypoints[descriptors_indices == 0]

# Level 0 - 4x4 grid
level0_1_4 = []
level0_5_8 = []
level0_9_12 = []
level0_13_16 = []
tot=0
for ini_i in range(0, 129, 128):
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

representation = np.hstack((0.25 * level2, 0.25 * level1_1, 0.25 * level1_2, 0.25 * level1_3, 0.25 * level1_4))
for g in range(0, 4):
    representation = np.hstack((representation, 0.5 * level0_1_4[g]))
    representation = np.hstack((representation, 0.5 * level0_5_8[g]))
    representation = np.hstack((representation, 0.5 * level0_9_12[g]))
    representation = np.hstack((representation, 0.5 * level0_13_16[g]))

v_words.append(representation)

max_value = 16581375 #255**3
interval = int(max_value / k)
col = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in col]

img = cv2.imread(train_images_filenames[0])

for i in range(0,100):
    #img[image_keypoints[i,0],image_keypoints[i,1]] = colors[image_predictions[i]]
    cv2.circle(img,(int(image_keypoints[i,0]),int(image_keypoints[i,1])),2 ,colors[image_predictions[i]])



plt.figure()
axes = plt.plot(level2)
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 2')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.figure()
plt.subplot(221)
axes = plt.plot(level1_1)
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 1.1')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(223)
axes = plt.plot(level1_2)
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 1.2')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(222)
axes = plt.plot(level1_3)
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 1.3')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(224)
axes = plt.plot(level1_4)
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 1.4')
plt.xlabel('visual word')
plt.ylabel('number of visual words')


plt.figure()
plt.subplot(441)
axes = plt.plot(level0_1_4[0])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.1')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(442)
axes = plt.plot(level0_1_4[1])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.2')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(443)
axes = plt.plot(level0_1_4[2])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.3')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(444)
axes = plt.plot(level0_1_4[3])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.4')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(445)
axes = plt.plot(level0_5_8[0])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.5')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(446)
axes = plt.plot(level0_5_8[1])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.6')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(447)
axes = plt.plot(level0_5_8[2])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.7')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(448)
axes = plt.plot(level0_5_8[3])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.8')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(449)
axes = plt.plot(level0_9_12[0])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.9')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,10)
axes = plt.plot(level0_9_12[1])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.10')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,11)
axes = plt.plot(level0_9_12[2])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.11')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,12)
axes = plt.plot(level0_9_12[2])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.12')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,13)
axes = plt.plot(level0_13_16[0])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.13')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,14)
axes = plt.plot(level0_13_16[1])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.14')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,15)
axes = plt.plot(level0_13_16[2])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.15')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.subplot(4,4,16)
axes = plt.plot(level0_13_16[3])
plt.xlim((0,512))
plt.ylim((0,3))
plt.title('level 0.16')
plt.xlabel('visual word')
plt.ylabel('number of visual words')

plt.show()
cv2.imshow('image',img)

k=cv2.waitKey(0)
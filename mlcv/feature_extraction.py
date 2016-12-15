import cv2
import numpy as np

import sys

def SIFT(gray):
    SIFT_detector = cv2.SIFT(nfeatures=100)
    kpt, des = SIFT_detector.detectAndCompute(gray, None)
    return kpt, des
def SURF(gray):
    SURF_detector = cv2.SURF(hessianThreshold=750, extended=1, upright=0)
    kpt, des = SURF_detector.detectAndCompute(gray, None)

    if len(kpt)>100:
        order=np.random.choice(len(des), 100)
        kp=np.empty([100, 128])
        desc=np.empty([100, 128])
        for index in np.sort(order):
            np.append(kp, kpt[index])
            np.append(desc, des[index])
        print kp
    else:
        kp=kpt
        desc=des
    return kp, desc
def ORB(gray):
    orb = cv2.ORB(nfeatures=100)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des
def BRISK(gray):
    brisk = cv2.BRISK()
    kp, des = brisk.detectAndCompute(gray, None)
    if len(kp) > 100:
        kp = kp[0:100]
        des = des[0:100]
    return kp, des

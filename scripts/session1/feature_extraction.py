import cv2
import numpy as np

import sys

def SIFT(gray, sigma, ct, et):
    SIFT_detector = cv2.SIFT(nfeatures=100, sigma=sigma, contrastThreshold=ct, edgeThreshold=et)
    kpt, des = SIFT_detector.detectAndCompute(gray, None)
    return kpt, des
def SURF(gray):
    SURF_detector = cv2.SURF(hessianThreshold=1500,nOctaves=2, extended=1, upright=1)
    kpt, des = SURF_detector.detectAndCompute(gray, None)

    return kpt, des
def ORB(gray):
    orb = cv2.ORB(nfeatures=250)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des
def BRISK(gray):
    brisk = cv2.BRISK(thresh=60)
    kp, des = brisk.detectAndCompute(gray, None)
    return kp, des
def BRIEF(gray):
    detector = cv2.SimpleBlobDetector()
    brief = cv2.DescriptorExtractor_create("BRIEF")
    kp = detector.detect(gray)
    kp, des = brief.compute(gray, kp)

    return kp, des

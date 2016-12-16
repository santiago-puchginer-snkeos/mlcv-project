import cv2
import numpy as np

import sys

def SIFT(gray, sigma, ct, et):
    SIFT_detector = cv2.SIFT(nfeatures=100, sigma=sigma, contrastThreshold=ct, edgeThreshold=et)
    kpt, des = SIFT_detector.detectAndCompute(gray, None)
    return kpt, des
def SURF(gray, sigma, ct, et):
    SURF_detector = cv2.SIFT(nfeatures=100, sigma=sigma, contrastThreshold=ct, edgeThreshold=et)
    #surfDetector = cv2.GridAdaptedFeatureDetector(SURF_detector, 100)
    keypoints = SURF_detector.detect(gray, None)
    SURF_detector = cv2.SURF(hessianThreshold=300, nOctaves=4, extended=1, upright=1)
    kpt, des = SURF_detector.compute(gray, keypoints=keypoints)

    return kpt, des
def ORB(gray, levels, et, wtak):
    orb = cv2.ORB(nfeatures=100, nlevels=levels, edgeThreshold=et, WTA_K=wtak)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des
def BRISK(gray):
    brisk = cv2.SIFT(nfeatures=100)
    keypoints= brisk.detect(gray, None)
    briskExtractor = cv2.DescriptorExtractor_create('BRISK')
    kp, des = briskExtractor.compute(gray, keypoints)
    return kp, des
def BRIEF(gray):
    detector = cv2.SIFT(nfeatures=100)
    brief = cv2.DescriptorExtractor_create("BRIEF")
    kp = detector.detect(gray)
    kp, des = brief.compute(gray, kp)

    return kp, des
def FREAK(gray, sigma, ct, et):
    surfDetector = cv2.SIFT(nfeatures=100, sigma=sigma, contrastThreshold=ct, edgeThreshold=et)
    keypoints = surfDetector.detect(gray, None)
    freakExtractor = cv2.DescriptorExtractor_create('FREAK')
    keypoints, descriptors = freakExtractor.compute(gray, keypoints)
    del freakExtractor
    return keypoints, descriptors
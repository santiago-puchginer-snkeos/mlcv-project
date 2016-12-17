import cPickle
import glob

accuracy_ORB = glob.glob("/home/adam/accuracy_SVMdef_ORB*.pickle")
accuracy_SIFT = glob.glob("/home/adam/accuracy_SVMdef_SIFT*.pickle")
time_ORB = glob.glob("/home/adam/time_SVMdef_ORB*.pickle")
time_SIFT = glob.glob("/home/adam/time_SVMdef_SIFT*.pickle")

accuracy_ORB_4levels_wtak2 = []
accuracy_ORB_4levels_wtak3 = []
accuracy_ORB_4levels_wtak4 = []

for accuracy in accuracy_ORB:
    if 'nlevels4_wtak2' in accuracy:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_ORB_4levels_wtak2.append(accuracy_value)
    elif 'nlevels4_wtak3' in accuracy:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_ORB_4levels_wtak3.append(accuracy_value)
    else:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_ORB_4levels_wtak4.append(accuracy_value)

accuracy_SIFT_sigma1_cT01 = []
accuracy_SIFT_sigma1_cT02 = []
accuracy_SIFT_sigma1_cT03 = []
accuracy_SIFT_sigma1_cT04 = []
accuracy_SIFT_sigma1_cT05 = []

for accuracy in accuracy_SIFT:
    if 'sigma1.0_cT0.01' in accuracy:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_SIFT_sigma1_cT01.append(accuracy_value)
    elif 'sigma1.0_cT0.02' in accuracy:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_SIFT_sigma1_cT02.append(accuracy_value)
    elif 'sigma1.0_cT0.03' in accuracy:
        with open(accuracy) as f:
            accuracy_value = cPickle.load(f)
        accuracy_SIFT_sigma1_cT03.append(accuracy_value)
    elif 'sigma1.0_cT0.04' in accuracy:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_SIFT_sigma1_cT04.append(accuracy_value)
    elif 'sigma1.0_cT0.05' in accuracy:
        with open(accuracy) as f:
           accuracy_value = cPickle.load(f)
        accuracy_SIFT_sigma1_cT05.append(accuracy_value)
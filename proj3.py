#!env/bin/python2

import os
import sys
import argparse
import itertools
import pickle

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import cv2
except ImportError as e:
    print("Could not load OpenCV: {}".format(e))

try:
    import numpy as np
except ImportError as e:
    print("Could not load Numpy: {}".format(e))

try:
    import sklearn
    from sklearn import svm
except ImportError as e:
    print("Could not load Scikit-Learn: {}".format(e))

def main(args):

    images = []

    if args.train:
        pairs = ((path, os.path.basename(os.path.dirname(path)))
                for path in image_paths(args.image))
        descriptors, classifications = get_classifications(pairs)

        test_model = svm.SVC()
        scores = sklearn.cross_validation.cross_val_score(
                test_model, descriptors, classifications, cv=10)
        print("Accuracy on 10-fold cross validation: {:0.2f} (+/- {:0.2f})" \
                .format(scores.mean(), scores.std() * 2))

        model = sklearn.svm.SVC()
        model.fit(descriptors, classifications)

        with open(args.model, 'w') as f:
            pickle.dump(model, f)

    else:
        with open(args.model, 'r') as f:
            model = pickle.load(f)

        for path in image_paths(args.image):
            im = cv2.imread(path, 0)
            classification = model.predict(get_descriptor(im))
            print("{} -> {}".format(path, classification))

def get_classifications(pairs):
    """Trains a SVM using each of the image/classification pairs."""

    descriptors = []
    classifications = []

    for path, classification in pairs:
        # Load the image.
        im = cv2.imread(path, 0)
        # Calculate the HOG descriptor.
        descriptors.append(get_descriptor(im))
        classifications.append(classification)

    return descriptors, classifications

def get_descriptor(im):
    return contourize(im)

def deskew(img, size=100, flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR):
    # Find the moments of the grayscale image.
    m = cv2.moments(img)

    # If it's already deskewed, do nothing.
    if abs(m['mu02']) < 1e-2:
        return img.copy()

    # Otherwise, compute the skew and apply it.
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*size*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(size, size),flags=flags)

    cv2.imshow('', img)
    cv2.waitKey(0)

    return img

def contourize(img, blur_size=5):
    blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
            cv2.THRESH_BINARY, 11, 2)

    image, contours, hierarchy = cv2.findContours(th,
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Put a bunch of properties in a vector.
    return [len(contours), sum(sum(th/255))]

def hog(img, num_bins=16):
    """Calculate the histogram of oriented gradients of the entire image.
    
    Taken in large part from
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_svm/py_svm_opencv/py_svm_opencv.html
    """
    # Take the oriented gradients with the Sobel image filter.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    # Turn it into polar coordinates.
    mag, ang = cv2.cartToPolar(gx, gy)

    # Construct a histogram.
    bins = np.int32(num_bins*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), num_bins)
            for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    # Hist is a 64-bit vector to be used as the descriptor.
    return hist

def image_paths(image_arg, match_extension='.jpg'):
    """Take a fuzzy list of image files or directories, and yield every file
    with the match_extension, if one is given."""
    for image_path in image_arg:
        if os.path.isfile(image_path):
            yield image_path
        else:
            for dirpath, _, files in os.walk(image_path):
                for f in files:
                    yield os.path.join(dirpath, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
            help="train the model and save it")
    parser.add_argument("--model", default="model.pkl",
            help="the file the model is saved to or loaded from")
    parser.add_argument("image", nargs="+",
            help="images or directory to classify")

    sys.exit(main(parser.parse_args()))

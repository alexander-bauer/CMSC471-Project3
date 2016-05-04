#!env/bin/python2

import os
import sys
import argparse
import itertools
import pickle

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    desc_arr = np.array(descriptors)

    ## For normalization with other kinds of descriptors
    #for colnum in range(desc_arr.shape[1]):
    #    desc_arr[:,colnum] /= max(desc_arr[:,colnum])

    return desc_arr, classifications

def get_descriptor(im):
    return np.hstack(im)/255

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

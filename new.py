##################################################
## Project: RotNIST
## Script purpose: To download MNIST dataset and append new rotated digits to it
## Date: 21st April 2018
## Author: Chaitanya Baweja, Imperial College London
##################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gzip
import os
import numpy as np
import tensorflow as tf
import time
import csv
from scipy import ndimage
from six.moves import urllib
from PIL import Image
from scipy.misc import imsave
#Url for downloading MNIST dataset
URL = 'http://yann.lecun.com/exdb/mnist/'
#Data Directory where all data is saved
DATA_DIRECTORY = "data"

# Params for MNIST

VALIDATION_SIZE = 5000  # Size of the validation set.

'''
Download the data from Yann's website, unless it's already here.
filename: filepath to images
Returns path to file
'''
def download(filename):
    #Check if directory exists
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    #Check if file exists, if not download
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
'''
Extract images from given file path into a 4D tensor [image index, y, x, channels].
Values are rescaled from [0, 255] down to [-0.5, 0.5].
filename: filepath to images
num: number of images
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_data(filename, num):
    print('Extracting', filename)
    #unzip data
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num, 28, 28) #reshape into tensor
    return data

'''
Extract the labels into a vector of int64 label IDs.
filename: filepath to labels
num: number of labels
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_labels(filename, num):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels

'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def expand_training_data(images, labels):

    expanded_images = []
    expanded_labels = []
    directory = os.path.dirname("data/New")
    if not tf.gfile.Exists("data/New"):
        tf.gfile.MakeDirs("data/New")
    k = 0 # counter
    for x, y in zip(images, labels):
        k = k+1
        if k%100==0:
            print ('expanding data : %03d / %03d' % (k,np.size(images,0)))
        print(x.shape)
        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        bg_value = -0.5 # this is regarded as background's value black

        image = np.reshape(x, (-1, 28))
        print(image.shape)
        #time.sleep(2)
        for i in range(4):
            # rotate the image with random degree
            angle = np.random.randint(-90,90,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = np.random.randint(-2, 2, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            #code for saving some of these for visualization purpose only
            image1 = (image*255) + (255 / 2.0)
            new_img1 = (new_img_*255) + (255 / 2.0)
            '''
            if k<50:
                NAME1 = DATA_DIRECTORY+"/New"+"/"+str(k)+"_0.jpeg"
                im = Image.fromarray(image1)
                im.convert('RGB').save(NAME1)
                im = Image.fromarray(new_img1)
                NAME = DATA_DIRECTORY+"/New"+"/"+str(k)+"_"+str(i+1)+".jpeg"
                im.convert('RGB').save(NAME)
            '''
            # register new training data

            expanded_images.append(new_img_)
            expanded_labels.append(y)


    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken

    expandedX=np.asarray(expanded_images)
    expandedY=np.asarray(expanded_labels)
    return expandedX, expandedY

# Prepare MNISt data
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    train_data_filename = download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename = download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)
    # Generate a validation set.
    print(test_data.shape)

    # Concatenate train_data & train_labels for random shuffle
    #if use_data_augmentation:
        #train_data,train_labels = expand_training_data(train_data, train_labels)
    #else:


    train_data,train_labels = expand_training_data(train_data, train_labels)
    if not os.path.isdir("data/train-images"):
        os.makedirs("data/train-images")
    if not os.path.isdir("data/test-images"):
        os.makedirs("data/test-images")
    # process train data
    with open("data/train-labels.csv", 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(train_data)):
            #imsave("data/train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
            writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])
    # repeat for test data
    with open("data/test-labels.csv", 'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',', quotechar='"')
        for i in range(len(test_data)):
            #imsave("mnist/test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
            writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])
    #return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels
prepare_MNIST_data(True)

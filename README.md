# RotNIST
Rotated MNIST dataset
This project was designed to create rotations of the popular MNIST digits(http://yann.lecun.com/exdb/mnist/).

The generative process used to generate the datasets is the following:
1) Pick a sample from the MNIST digit recognition dataset;
2) Create a rotated version of the sample. The digits were rotated by an angle generated randomly between -\pi and \pi radians.
3) Go back to 1 until enough samples(by default 4, can be passed as argument) are generated.

## INSTALLATION OF REQUIRED TOOLS
#### 1. Tensorflow
Refer to the following link https://www.tensorflow.org/install/install_sources. Tensorflow is used as backend for Keras. The link contains installation instructions with and without gpu support

#### 2.Skimage
Scikit-image is an image processing toolbox for SciPy. It is used for loading,saving and applying various transformations like color to gray and gray to color on images.

* Refer following link for installation instructions http://scikit-image.org/docs/dev/install.html

#### 3. numpy
No introductions required!!!
* To install numpy

    sudo pip install numpy

#### 4. PIL
Pillow is the friendly PIL fork by Alex Clark and Contributors. PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
* To install PIL

    sudo pip install pillow

#### 5. Scipy
SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering.
* To install Scipy

    sudo pip install scipy

#### 6. SIX
Six is a Python 2 and 3 compatibility library. It provides utility functions for smoothing over the differences between the Python versions with the goal of writing Python code that is compatible on both Python versions.
* To install SIX

    sudo pip install six

# Quick start

Run the following commands in Terminal.

git clone https://github.com/ChaitanyaBaweja/RotNIST.git

cd RotNIST

python main.py

python images.py

* main.py Contains the code to save rotated images in 4D numpy arrays and return them for other codes, to be used as auxiliary file for networks
* images.py Contains the code to save rotated images as .jpg files

# Soon to come
* Wrapper function to put all these together and pass arguments
* Faster code implementation
* Results of this data set on Capsule networks

## Contact Details:
Chaitanya Baweja : chaitanya.baweja17@imperial.ac.uk

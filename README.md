# RotNIST
Rotated MNIST dataset
This project was designed to create rotations of the popular MNIST digits(http://yann.lecun.com/exdb/mnist/).

The generative process used to generate the datasets is the following:
1) Pick sample **x,y** \in **X** from the MNIST digit recognition dataset;
2) Create a rotated version x`  of **x**. the digits were rotated by an angle generated uniformly between 0 and $2 \pi$ radians.
3) Go back to 1 until enough samples are generated.

mnist-rot:  Thus the factors of variation are the rotation angle and the factors of variation already contained in MNIST, such as handwriting style:

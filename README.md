# RotNIST
Rotated MNIST dataset
This project was designed to create rotations of the popular MNIST digits(http://yann.lecun.com/exdb/mnist/).

The generative process used to generate the datasets is the following:
1) Pick sample **x,y** \in **X** from the MNIST digit recognition dataset;
2) Create a rotated version **x^{\prime}**  of **x**. The digits were rotated by an angle generated uniformly between -\pi and \pi radians.
3) Go back to 1 until enough samples are generated.

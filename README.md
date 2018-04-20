# RotNIST
Rotated MNIST dataset
This project was designed to create rotations of the popular MNIST digits(http://yann.lecun.com/exdb/mnist/).

The generative process used to generate the datasets is the following:
1) Pick sample $(x,y) \in {\cal X}$ from the MNIST digit recognition dataset (http://yann.lecun.com/exdb/mnist/);
2) Create a perturbed version $\widehat{x}$ of $x$ according to some factors of variation;
3) Add $(\widehat{x},y)$ to new dataset $\widehat{{\cal X}}$;
4) Go back to 1 until enough samples are generated.

mnist-rot: the digits were rotated by an angle generated uniformly between 0 and $2 \pi$ radians. Thus the factors of variation are the rotation angle and the factors of variation already contained in MNIST, such as handwriting style:

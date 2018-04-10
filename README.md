Galaxies - a CUDA project
=========================

PV197 GPU Programming course, Masaryk University, Autumn 2017

Compilation
-----------

`nvcc framework.cu -o framework`

Assignment
----------

See [project-announcement.pdf](https://github.com/rtrembecky/galaxies/blob/master/project-announcement.pdf) for
full assignment (and better formatting).

The project is focused on computing similarity of galaxies. Your task is to
implement a CUDA version of an algorithm, which compares distances of all pairs of
stars in two representations of the same galaxy. The galaxy is represented as a
vector of stars, each star has Cartesian coordinates in 3D space. The algorithm
needs to evaluate the following formula:

dist = sqrt( 1/(n(n-1)) * sum_{i=1}^{n-1} sum_{j=i+1}^{n} (d_{ij}^A -
d_{ij}^B)^2 )

where n is number of stars and d_{ij}^A is Euclidean distance between i-th star and
j-th star in galaxy A. Thus, the formula computes differences
of all-to-all distances, subtracts them, squares them and sums them and
normalizes the number when summed to not be influenced by the number of stars.

If you are not sure if you understand what to implement, see kernel_CPU.C - the
C equivalent of what you are required to implement is there.

In study materials, there is a framework available. It prepares all input data,
copies them to GPU, performs CPU computation, benchmarks CPU and GPU
implementation and check, if data computed at GPU and CPU matches. Your task is
to write a GPU code (into kernel.cu). There are several technical rules for the
implementation:
- single-precision (e.g. float) is enough for this project
- the input size can be any number fitting into GPU memory
- the code must run on CUDA card with compute capability 3.0 or newer
- the performance will be measured using 2 000 stars and 50 000 stars per galaxy
- the highest acceptable error of GPU version is 1%
- your code is expected to be written in kernel.cu -- it must run with the
original framework (of course you can change your framework for debugging
purposes)
- this is individual project, any form of cooperation is strictly prohibited.
# Othogonal Polynomials on Bubble-Dimaond Fractals

Elena P. Axinn, Gamal Mograby, Calvin Osborne, Kasso A. Okoudjou, Olivia Rigatti, Helen Shi

Implemented by Calvin Osborne.

The code provided here generated the figures from this paper. We follow the computations laid out in the paper to compute the harmonic basis, monomial basis, and Legendre polynomials on a graph approximation of the bubble-diamond fractal with a given branching parameter and number of layers.

### Plotting

To plot a particular graph, see the last section of the code. In particular, the harmonic basis ``fb``, the monomial basis ``Pb``, and the Legendre polynomials ``ob`` can all be plotted.

By default, the first five Legendre polynomials are plotted on a bubble-diamond graph with branching parameter $b = 2$ and $m = 5$ layers of approximation.

### Config

To make changes to the underlying bubble-diamond graph, use the ``config.ini`` file.
* ``b``: the branching parameter.
* ``m``: the number of layers of the graph approximation.
* ``j``: the number of Legendre polynomials to generate.
* ``ell``: the height from the unit interval to the furthest branch of $V_1$.
* ``display_mode``: either ``2D``, which displays just the underlying bubble-diamond graph, or ``3D``, which displays the desired polynomial functions on the underlying bubble-diamond graph.

### Requirements

Python Packages: `configparser`, `matplotlib`, `numpy`, `scipy`.

Version Information:
* `configparser`: 7.1.0
* `matplotlib`: 3.9.2
* `numpy`: 2.1.1
* `scipy`: 1.14.1

### Licensing

See `LICENSE` for information about licensing this code.

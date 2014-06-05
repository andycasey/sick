# coding: utf-8

""" Convenient plotting functions """

from __future__ import division, print_function

__author__ = ("Triangle.py (corner) was written by Dan Foreman-Mackey" 
    "Andy Casey <arc@ast.cam.ac.uk> wrote the other plotting functions to "
    "match the feel of triangle.py")

__all__ = ["chains", "corner"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from triangle import corner

def chains(xs, labels=None, truths=None, truth_color=u"#4682b4", burn_in=None,
    alpha=0.5, fig=None):
    """
    Create a plot showing the walker values for each dimension at every step.

    Args:
        xs (array_like) : The samples. This should be a 3D array of size 
            (n_walkers, n_steps, n_dimensions)

        labels (iterable, optional) : A list of names for the dimensions.

        truths (iterable, optional) : A list of reference values to indicate on
            the plots.

        truth_color (str, optional) : A `matplotlib` style color for the `truths`
            markers.

        burn_in (int, optional) : A reference step to indicate on the plots.

        alpha (float between [0, 1], optional) : Transparency of individual walker
            lines.

        fig (`matplotlib.Figure`, optional) : Overplot onto the provided figure object.
    
    Returns:
        A `matplotlib.Figure` object.
    """

    n_walkers, n_steps, K = xs.shape

    if labels is not None:
        assert len(labels) == K

    if truths is not None:
        assert len(truths) == K

    factor = 2.0
    lbdim = 0.5 * factor
    trdim = 0.2 * factor
    whspace = 0.10
    width = 8.
    height = factor*K + factor * (K - 1.) * whspace
    dimy = lbdim + height + trdim
    dimx = lbdim + width + trdim

    if fig is None:
        fig, axes = plt.subplots(K, 1, figsize=(dimx, dimy))

    else:
        try:
            axes = np.array(fig.axes).reshape((1, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                "dimensions K={1}".format(len(fig.axes), K))

    lm = lbdim / dimx
    bm = lbdim / dimy
    trm = (lbdim + height) / dimy
    fig.subplots_adjust(left=lm, bottom=bm, right=trm, top=trm,
        wspace=whspace, hspace=whspace)

    for k, ax in enumerate(axes):

        for walker in range(n_walkers):
            ax.plot(xs[walker, :, k], color="k", alpha=alpha)

        if burn_in is not None:
            ax.axvline(burn_in, color="k", marker=":")

        if truths is not None:
            ax.axhline(truths[k], color=truth_color, lw=2)

        ax.set_xlim(0, n_steps)
        if k < K - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Step")

        ax.yaxis.set_major_locator(MaxNLocator(4))
        [l.set_rotation(45) for l in ax.get_yticklabels()]
        if labels is not None:
            ax.set_ylabel(labels[k])
            ax.yaxis.set_label_coords(-0.05, 0.5)

    return fig


#def projection(xs, lnprob, model, data, num=100):

    
    # 
# Mean acceptance fracs?


# coding: utf-8

""" Convenient plotting functions """

from __future__ import division, print_function

__author__ = ("Triangle.py (corner) was written by Dan Foreman-Mackey, and " 
    "Andy Casey wrote the other plotting functions to match the (beautiful)"
    "look of triangle.py")

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
            ax.axvline(burn_in, color="k", linestyle=":")

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


# Can do it from: data, model, sampler, n
def projection(sampler, model, data, n=100, fig=None):

    if not isinstance(data, (tuple, list)) or \
    any([not isinstance(each, sick.specutils.Spectrum1D) for each in data]):
        raise TypeError("Data must be a list-type of Spectrum1D objects.")

    K = len(data)

    factor = 3.0
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

    # Find the most probable sampled theta and compute spectra for it
    max_lnprob_index = np.argmax(sampler.flatlnprobability)
    max_lnprob_theta = sampler.flatchain[max_lnprob_index]
    max_lnprob_fluxes = model(observations=data, **dict(zip(model.dimensions, max_lnprob_theta)))

    if n > 0:
        # Draw samples from sampler.chain and compute spectra for them
        sampled_fluxes = []
        n_samples = len(sampler.flatchain)

        for i in range(n):
            sampled_theta = dict(zip(model.dimensions, sampler.flatchain[np.random.randint(0, n_samples)]))
            try:
                sampler_flux = model(observations=data, **sampled_theta)
            except:
                continue
            else:
                sampled_fluxes.append(sampler_flux)

    for k, (ax, max_lnprob_flux, observed_spectrum) in enumerate(zip(axes, max_lnprob_fluxes, data)):

        # Draw the random samples from the chain
        if n > 0:
            for sampled_flux in sampled_fluxes:
                ax.plot(observed_spectrum.disp, sampled_flux[k], color="#666666")

        # Draw the ML spectra
        ax.plot(observed_spectrum.disp, max_lnprob_flux, color="r", lw=2)

        # Plot the data
        ax.plot(observed_spectrum.disp, observed_spectrum.flux, color="k")

        # By default only show common overlap between the model and spectral data
        if extents is None:
            finite_data = np.isfinite(observed_spectrum.flux)
            finite_model = np.isfinite(model_flux)

            x_extent = [
                np.max([observed_spectrum.disp[indices][0]  for indices in (finite_model, finite_data)]),
                np.min([observed_spectrum.disp[indices][-1] for indices in (finite_model, finite_data)]),
            ]

            indices = observed_spectrum.disp.searchsorted(x_extent)
            y_extent = [
                0.9 * np.min(observed_aperture.flux[indices[0]:indices[1]]),
                1.1 * np.max(observed_aperture.flux[indices[0]:indices[1]])
            ]
            ax.set_xlim(x_extent)
            ax.set_ylim(y_extent)

        else:
            ax.set_xlim(extents[k][0])
            ax.set_ylim(extents[k][1])

        # Labels and ticks
        if k < K - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Step")
        ax.set_ylabel("Flux, $F_\lambda$")
        ax.yaxis.set_label_coords(-0.05, 0.5)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        [l.set_rotation(45) for l in ax.get_yticklabels()]

    return fig


# coding: utf-8

""" Convenient plotting functions """

from __future__ import division, print_function

__author__ = ("Triangle.py (corner) was written by Dan Foreman-Mackey, and " 
    "Andy Casey wrote the other plotting functions to match the (beautiful)"
    "look of triangle.py")

__all__ = ["chains", "corner", "projection"]

import numpy as np
import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from triangle import corner

import specutils

# Update the triangle.corner docstring to be sphinxy
corner.__doc__ = """
    [This function was written by Dan Foreman-Mackey as part of ``triangle.py``.
    Please see the associated ``LICENSE`` file for ``triangle.py``, which is 
    also available from https://github.com/dfm/triangle.py/blob/master/LICENSE]

    Make a *sick* corner plot showing the projections of a data set in a
    multi-dimensional space. kwargs are passed to hist2d() or used for
    ``matplotlib`` styling.

    :param xs:
        The samples. This should be a 1- or 2-dimensional array. For a 1-D
        array this results in a simple histogram. For a 2-D array, the zeroth
        axis is the list of samples and the next axis are the dimensions of
        the space.

    :type xs:
        array_like ``(nsamples, ndim)``

    :param weights: [optional]
        The weight of each sample. If ``None`` (default), samples are given
        equal weight.

    :type weights:
        array_like ``(nsamples,)``

    :param labels: [optional]
        A list of names for the dimensions.

    :type labels:
        iterable ``(ndim,)``

    :param show_titles: [optional]
        Displays a title above each 1-D histogram showing the 0.5 quantile
        with the upper and lower errors supplied by the quantiles argument.

    :type show_titles:
        bool

    :param title_fmt: [optional]
        The format string for the quantiles given in titles.
        (default: ``.2f``)

    :type title_fmt:
        str

    :param title_args: [optional]
        Any extra keyword arguments to send to the ``add_title`` command.

    :type title_args:
        dict

    :param extents: [optional]
        A list where each element is either a length 2 tuple containing
        lower and upper bounds (extents) or a float in range (0., 1.)
        giving the fraction of samples to include in bounds, e.g.,
        [(0.,10.), (1.,5), 0.999, etc.].
        If a fraction, the bounds are chosen to be equal-tailed.

    :type extents:
        iterable ``(ndim,)``

    :param truths: [optional]
        A list of reference values to indicate on the plots.

    :type truths:
        iterable ``(ndim,)``

    :param truth_color: [optional]
        A ``matplotlib`` style color for the ``truths`` makers.

    :type truth_color:
        str

    :param scale_hist: [optional]
        Should the 1-D histograms be scaled in such a way that the zero line
        is visible?

    :type scale_hist:
        bool

    :param quantiles: [optional]
        A list of fractional quantiles to show on the 1-D histograms as
        vertical dashed lines.

    :type quantiles:
        iterable

    :param verbose: [optional]
        If true, print the values of the computed quantiles.

    :type verbose:
        bool

    :param plot_contours: [optional]
        Draw contours for dense regions of the plot.

    :type verbose:
        bool

    :param plot_datapoints: [optional]
        Draw the individual data points.

    :type plot_datapoints:
        bool

    :param fig: [optional]
        Overplot onto the provided figure object.

    :type fig:
        ``matplotlib.Figure``

    :raises ValueError:
        If a ``fig`` is provided with the incorrect number of axes.

    :returns:
        The triangle figure.

    :rtype:
        :class:`matplotlib.Figure`
"""

def chains(xs, labels=None, truths=None, truth_color=u"#4682b4", burn_in=None,
    alpha=0.5, fig=None):
    """
    Create a plot showing the walker values for each parameter at every step.

    :param xs:
        The samples. This should be a 3D :class:`numpy.ndarray` of size (``n_walkers``,
        ``n_steps``, ``n_parameters``).

    :type xs:
        :class:`numpy.ndarray`

    :param labels: [optional]
        Labels for all the parameters.

    :type labels:
        iterable of strings or None

    :param truths: [optional]
        Reference values to indicate on the plots.

    :type truths:
        iterable of floats or None

    :param truth_color: [optional]
        A ``matplotlib`` style color for the ``truths`` markers.

    :param burn_in: [optional]
        Reference step to indicate on the plots.

    :type burn_in:
        integer or None

    :param alpha: [optional]
        Transparency of individual walker lines between zero and one.

    :type alpha:
        float

    :param fig: [optional]
        Overplot onto the provided figure object.

    :type fig:
        :class:`matplotlib.Figure` or None
    
    :raises ValueError:
        If a ``fig`` is provided with the incorrect number of axes.

    :returns:
        The chain figure.

    :rtype:
        :class:`matplotlib.Figure`
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
    width = 15.
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
                "parameters K={1}".format(len(fig.axes), K))

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


def projection(sampler, model, data, n=100, extents=None, fig=None, figsize=None):
    """
    Project the maximum likelihood values and sampled posterior points as spectra.

    :param sampler:
        The sampler employed.

    :type sampler:
        :class:`emcee.EnsembleSampler`

    :param model:
        The model employed.

    :type model:
        :class:`sick.models.Model`

    :param data:
        The observed spectra.

    :type data:
        iterable of :class:`sick.specutils.Spectrum1D` objects

    :param extents: [optional]
        The wavelength extents to plot for each channel in the form of [(min_chan_1,
        max_chan_1), ..., (min_chan_N, max_chan_N)]
    
    :type extents:
        tuple or None

    :param fig: [optional]
        Overplot onto the provided figure object.

    :type fig:
        :class:`matplotlib.Figure` or None
    
    :param figsize: [optional]
        The figure size (x-dimension, y-dimension) in inches.

    :type figsize:
        tuple or None

    :raises ValueError:
        If a ``fig`` is provided with the incorrect number of axes.

    :raise TypeError:
        If the ``data`` are not provided in the correct type.

    :returns:
        The projection figure.

    :rtype:
        :class:`maplotlib.Figure`
    """
    if not isinstance(data, (tuple, list)) or \
    any([not isinstance(each, specutils.Spectrum1D) for each in data]):
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

    if figsize is None:
        figsize = (dimx, dimy)
    if fig is None:
        fig, axes = plt.subplots(K, 1, figsize=figsize)

    else:
        try:
            axes = np.array(fig.axes).reshape((1, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                "parameters K={1}".format(len(fig.axes), K))

    # Find the most probable sampled theta and compute spectra for it
    max_lnprob_index = np.argmax(sampler.lnprobability.flatten())
    max_lnprob_theta = sampler.flatchain[max_lnprob_index]
    max_lnprob_fluxes = model(observations=data, **dict(zip(model.parameters, max_lnprob_theta)))

    if n > 0:
        # Draw samples from sampler.chain and compute spectra for them
        sampled_fluxes = []
        n_samples = len(sampler.flatchain)

        for i in range(n):
            sampled_theta = dict(zip(model.parameters, sampler.flatchain[np.random.randint(0, n_samples)]))
            try:
                sampler_flux = model(observations=data, **sampled_theta)
            except:
                continue
            else:
                sampled_fluxes.append(sampler_flux)
    
    if len(data) == 1:
        axes = [axes]

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
            finite_model = np.isfinite(max_lnprob_flux)

            x_extent = [
                np.max([observed_spectrum.disp[indices][0]  for indices in (finite_model, finite_data)]),
                np.min([observed_spectrum.disp[indices][-1] for indices in (finite_model, finite_data)]),
            ]

            indices = observed_spectrum.disp.searchsorted(x_extent)
            finite_observed_flux = observed_spectrum.flux[indices[0]:indices[1]]
            y_extent = [
                0.9 * np.min(finite_observed_flux[np.isfinite(finite_observed_flux)]),
                1.1 * np.max(finite_observed_flux[np.isfinite(finite_observed_flux)])
            ]
            ax.set_xlim(x_extent)
            ax.set_ylim(y_extent)

        else:
            ax.set_xlim(extents[k][0])
            ax.set_ylim(extents[k][1])

        # Labels and ticks
        if not (k < K - 1):
            ax.set_xlabel("Wavelength, $\lambda$ ($\AA$)")

        ax.set_ylabel("Flux, $F_\lambda$")
        ax.yaxis.set_label_coords(-0.05, 0.5)

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        [l.set_rotation(45) for l in ax.get_yticklabels()]

    return fig


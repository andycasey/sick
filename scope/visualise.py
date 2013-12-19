# coding: utf-8

""" Visualising output from SCOPE analyses. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard libraries
import logging
import pickle

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import triangle

__all__ = ["plot_all_results", "plot_result"]

def plot_result(posteriors, state, observed_spectra, model_spectra, masks):
    """ Generates diagnostic plots for the SCOPE results and returns the figure canvas. """

    mask_colors = ["w", "#aa5f5f", "w", "#acacac"]

    spectrum_figure = plt.figure()
    num_spectra = len(observed_spectra)

    axis = spectrum_figure.add_subplot(num_spectra + 1, 1, 1)
    axis.axis('off')
    axis.set_title('$\chi^2$ = {chi_sq:.2f}, $N_{{dof}}$ = {num_dof:.0f}'
        .format(**state))
 
    posteriors_formatted = "\n".join(["{parameter}:  {value:.3f}".format(parameter=parameter, value=posteriors[parameter])
        for parameter in sorted(posteriors.keys(), key=len)])

    axis.text(0.05, 0.05, posteriors_formatted, fontsize='x-small', transform=axis.transAxes)

    for i, (observed_spectrum, model_spectrum, mask) in enumerate(zip(observed_spectra, model_spectra, masks), start=2):
        axis = spectrum_figure.add_subplot(num_spectra + 1, 1, i)

        axis.plot(model_spectrum.disp, model_spectrum.flux, 'b', zorder=1)
        axis.plot(observed_spectrum.disp, observed_spectrum.flux, 'k', zorder=5)

        colors = [mask_colors[int(pixel + 2)] for pixel in mask]
        axis.scatter(model_spectrum.disp, [1.1] * len(mask), marker='|', edgecolors=colors, linewidths=1, s=100, zorder=1)

        # Get the overlap
        xlim = [
            max(observed_spectrum.disp[0], model_spectrum.disp[0]),
            min(observed_spectrum.disp[-1], model_spectrum.disp[-1]) 
        ]
        axis.set_xlim(xlim)
        axis.set_ylim(0, 1.2)

        axis.set_ylabel('Normalised Flux')
        if i == num_spectra + 1:
            axis.set_xlabel('Wavelength (${\AA}$)')

    # Was this a MCMC result?
    if "log_likelihood" in state:
        progress_figure = plt.figure()
        acceptance_fraction_axes = progress_figure.add_subplot(311)

        x = np.arange(1, len(state["mean_acceptance_fractions"]) + 1)
        acceptance_fraction_axes.plot(x, state["mean_acceptance_fractions"], c="k", lw=2)
        acceptance_fraction_axes.set_xlabel("Step")
        acceptance_fraction_axes.set_ylabel("Mean acceptance fraction")
        acceptance_fraction_axes.set_xlim(1, x[-1])

        log_likelihood_axes = progress_figure.add_subplot(312)

        x = np.arange(1, len(state["sampled_parameters"]) + 1)
        log_likelihood_axes.plot(x, state["sampled_parameters"][:, -1], c="k", lw=2)
        log_likelihood_axes.set_xlabel("Sampling")
        log_likelihood_axes.set_ylabel("log(L)")
        log_likelihood_axes.set_xlim(1, x[-1])

        chi_sq_axes = progress_figure.add_subplot(313)

        chi_sq_axes.plot(x, state["sampled_parameters"][:, -2], c="k", lw=2)
        chi_sq_axes.set_xlabel("Sampling")
        chi_sq_axes.set_ylabel("$\chi^2$")
        chi_sq_axes.set_xlim(1, x[-1])

        # All sampled points
        fig_sampled_points = plt.figure()
        num_parameters = len(posteriors)
        for i, parameter in enumerate(posteriors.keys()):
            y = state["sampled_parameters"][:, i]
            axes = fig_sampled_points.add_subplot(num_parameters, 1, i + 1)
            axes.plot(x, y, c="k", lw=1)
            axes.set_ylabel(parameter)
            axes.set_xlim(1, x[-1])

        axes.set_xlabel("Sampling")

        return
        # Produce a corner plot, because they're beautiful *and* useful
        corner_figure = triangle.corner(state["sampled_parameters"][:, :-2], labels=posteriors.keys())

        return [spectrum_figure, progress_figure, corner_figure]

    return [spectrum_figure]


def plot_saved_result(pickled_result):
    """ Loads a pickled result and generates diagnostic plots. """

    with open(pickled_result, "r") as fp:
        results = pickle.load(fp)

    return plot_result(*results)


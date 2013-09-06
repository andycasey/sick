# coding: utf-8

""" SCOPE tests to use for API development and testing. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Third-party
import matplotlib.pyplot as plt
import numpy as np


def animate_setup(num_spectra):
    """Creates a figure and axes for the animate_callback() function
    to be run after each analysis iteration.

    Inputs
    ------

    num_spectra : int
        The number of observed spectral apertures for this star.
    """

    fig = plt.figure()

    # Create axes
    for i in xrange(num_spectra):
        ax = fig.add_subplot(num_spectra, 1, i + 1)
        # One for model, and one for observed
        ax.plot([], [], 'k', zorder=5)
        ax.plot([], [], 'b', zorder=1)

    return fig


def animate_callback(total_chi_sq, num_dof, parameters, observed_spectra, model_spectra):
    """Callback function that can be used during each iteration of analysis."""

    # Get the current figure
    fig = plt.gcf()
    
    # Plot the relevant spectra on each axes
    for ax, observed_spectrum, model_spectrum in zip(fig.axes, observed_spectra, model_spectra):

        # Calculate the limits
        xlims = [observed_spectrum.disp[0], observed_spectrum.disp[-1]]
        ylims = [0, np.max(observed_spectrum.flux)]

        # Update the model and observed data
        ax.lines[0].set_data(np.array([observed_spectrum.disp, observed_spectrum.flux]))
        ax.lines[1].set_data(np.array([model_spectrum.disp, model_spectrum.flux]))

        # Update the limits
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    # Ensure everything is drawn
    plt.draw()
    plt.pause(0.01)
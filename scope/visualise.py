# coding: utf-8

""" SCOPE tests to use for API development and testing. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard libraries
import logging

# Third-party
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['animate_setup', 'animate_callback', 'plot_all_results', 'plot_result']

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


def plot_result(chi_sq, num_dof, posteriors, observed_spectra, model_spectra):
    """Plots the SCOPE result and returns the figure canvas."""

    if posteriors is None:
        return None

    fig = plt.figure()
    num_spectra = len(observed_spectra)

    axis = fig.add_subplot(num_spectra + 1, 1, 1)
    axis.axis('off')
    axis.set_title('$\chi^2$ = {chi_sq:.2f}, $N_{{dof}}$ = {num_dof:.0f}'
        .format(chi_sq=chi_sq, num_dof=num_dof))
 
    posteriors_formatted = "\n".join(["{parameter}:  {value:.3f}".format(parameter=parameter, value=posteriors[parameter])
        for parameter in sorted(posteriors.keys(), key=len)])

    axis.text(0.05, 0.05, posteriors_formatted, fontsize='x-small', transform=axis.transAxes)

    for i, (observed_spectrum, model_spectrum) in enumerate(zip(observed_spectra, model_spectra), start=2):
        axis = fig.add_subplot(num_spectra + 1, 1, i)

        axis.plot(model_spectrum.disp, model_spectrum.flux, 'b', zorder=1)
        axis.plot(observed_spectrum.disp, observed_spectrum.flux, 'k', zorder=5)

        # Get the overlap
        xlim = [
            max(observed_spectrum.disp[0], model_spectrum.disp[np.where(np.isfinite(model_spectrum.flux) == True)[0][0]]),
            min(observed_spectrum.disp[-1], model_spectrum.disp[np.where(np.isfinite(model_spectrum.flux) == True)[0][-1]])
        ]
        axis.set_xlim(xlim)
        axis.set_ylim(0, 1.2)

        axis.set_ylabel('Normalised Flux')
        if i == num_spectra + 1:
            axis.set_xlabel('Wavelength (${\AA}$)')
        

    return fig


def plot_all_results(results, output_prefix):
    """Plots and saves each result to file."""

    for i, result in enumerate(results, start=1):
        
        fig = plot_result(*result)
        if fig is None: continue

        filename = '{output_prefix}-{i}.png'.format(output_prefix=output_prefix, i=i)
        logging.info("Saved plotted result to {filename}".format(filename=filename))

        plt.savefig(filename)
        plt.close('all')

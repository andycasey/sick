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
        ax.plot([], [], 'k', zorder=10)
        ax.plot([], [], 'b', zorder=5)

        # One for the mask
        ax.scatter([], [], marker='|', linewidths=1, s=100, zorder=1)

    return fig


def animate_callback(total_chi_sq, num_dof, parameters, observed_spectra, model_spectra, masks):
    """Callback function that can be used during each iteration of analysis."""

    # Get the current figure
    fig = plt.gcf()
    
    # Mask color order:
    #> -2: Not interested in this region, and it was non-finite (not used).
    #> -1: Interested in this region, but it was non-finite (not used).
    #>  0: Not interested in this region, it was finite (not used).
    #>  1: Interested in this region, it was finite (used for \chi^2 determination)
    mask_colors = ["w", "#aa5f5f", "w", "#acacac"]

    # Plot the relevant spectra on each axes
    for ax, observed_spectrum, model_spectrum, mask in zip(fig.axes, observed_spectra, model_spectra, masks):

        # Calculate the limits
        xlims = [observed_spectrum.disp[0], observed_spectrum.disp[-1]]
        ylims = [0, np.max(list(observed_spectrum.flux) + [1.20])]

        # Update the model and observed data
        ax.lines[0].set_data(np.array([observed_spectrum.disp, observed_spectrum.flux]))
        ax.lines[1].set_data(np.array([model_spectrum.disp, model_spectrum.flux]))

        if len(ax.collections[0].get_offsets()) == 0:
            ax.collections[0].set_offsets(np.vstack([model_spectrum.disp, [1.10] * len(mask)]).T)

        colors = [mask_colors[int(i + 2)] for i in mask]
        ax.collections[0].set_edgecolors(colors)

        # Update the limits
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    # Ensure everything is drawn
    plt.draw()
    plt.pause(0.005)


def plot_mcmc_result(posteriors, maximum_log_likelihood, minimum_chi_sq, num_dof, mean_acceptance_fractions,
    observed_spectra, model_spectra, masks, pos, lnprob, state, sampled_data):
    """Plots the results from a MCMC simulation and returns the figure canvas."""

    # posteriors: dict with parameters/best values as key/values
    # log_likelihood: float of maximum likelihood
    # minimum_chi_sq
    # mean_acceptance_fractions: list of floats of length NSTEPS showing the mean acceptance fraction for each step
    # observed_spectra: list of specutils.Spectrum1D
    # model_spectra : list of specutils.Spectrum1D
    # list (length same as model/observed_spectra) of arrays where the values correspond to each flux pixel:
        #> -2: Not interested in this region, and it was non-finite (not used).
        #> -1: Interested in this region, but it was non-finite (not used).
        #>  0: Not interested in this region, it was finite (not used).
        #>  1: Interested in this region, it was finite (used for \chi^2 determination)
    # pos, lnprob, state of the data
    # sampled_data: an array of length (N, M+2) where M is length of posteriors (plus chi_sq, log_likelihood), N is the number of finite samples

    mask_colors = ["w", "#aa5f5f", "w", "#acacac"]

    # Figures we want:
    # 1) mean acceptance fraction over steps
    # 1.1) log likelihood for each finite sampling
    # 1.2) chi-squared for each finite sampling
    # 2) model spectra/observed spectra with masks
    # 3) triangle plot for all sampled parameters.
    # 4) plot of all parameters for all finite samplings
    fig = plt.figure()
    acceptance_fraction_axes = fig.add_subplot(311)

    x = np.arange(1, len(mean_acceptance_fractions) + 1)
    acceptance_fraction_axes.plot(x, mean_acceptance_fractions, c="k", lw=2)
    acceptance_fraction_axes.set_xlabel("Step")
    acceptance_fraction_axes.set_ylabel("Mean acceptance fraction")
    acceptance_fraction_axes.set_xlim(1, x[-1])

    log_likelihood_axes = fig.add_subplot(312)

    x = np.arange(1, len(sampled_data) + 1)
    log_likelihood_axes.plot(x, sampled_data[:, -1], c="k", lw=2)
    log_likelihood_axes.set_xlabel("Sampling")
    log_likelihood_axes.set_ylabel("log(L)")
    log_likelihood_axes.set_xlim(1, x[-1])

    chi_sq_axes = fig.add_subplot(313)

    chi_sq_axes.plot(x, sampled_data[:, -2], c="k", lw=2)
    chi_sq_axes.set_xlabel("Sampling")
    chi_sq_axes.set_ylabel("$\chi^2$")
    chi_sq_axes.set_xlim(1, x[-1])
    
    # All sampled points
    fig_sampled_points = plt.figure()
    num_parameters = len(posteriors)
    for i, parameter in enumerate(posteriors.keys()):
        y = sampled_data[:, i]
        axes = fig_sampled_points.add_subplot(num_parameters, 1, i + 1)
        axes.plot(x, y, c="k", lw=1)
        axes.set_ylabel(parameter)
        axes.set_xlim(1, x[-1])

    axes.set_xlabel("Sampling")

    fig2 = plt.figure()
    axes = [fig2.add_subplot(len(observed_spectra), 1, i) for i in xrange(1, len(observed_spectra) + 1)]
    for axis, observed_spectrum, model_spectrum, mask in zip(axes, observed_spectra, model_spectra, masks):

        # Model spectra
        axis.plot(model_spectrum.disp, model_spectrum.flux, c="b")
        axis.plot(observed_spectrum.disp, observed_spectrum.flux, "k")

        colors = [mask_colors[int(pixel + 2)] for pixel in mask]
        axis.scatter(model_spectrum.disp, [1.1] * len(mask), marker='|', edgecolors=colors, linewidths=1, s=100, zorder=1)

        xlim = [max(observed_spectrum.disp[0], model_spectrum.disp[0]), min(observed_spectrum.disp[-1], model_spectrum.disp[-1])]
        axis.set_xlim(xlim)
        axis.set_ylim(0, 1.2)
        axis.set_ylabel("Flux")

    axis.set_xlabel("Wavelength")
    axes[0].set_title(",".join(["{0}:{1:.2e}".format(key, value) for key, value in posteriors.iteritems()]), fontsize=7)

    return (fig, fig2, fig_sampled_points)


def plot_result(chi_sq, num_dof, posteriors, observed_spectra, model_spectra, masks):
    """Plots the SCOPE result and returns the figure canvas."""

    if posteriors is None:
        return None

    mask_colors = ["w", "#aa5f5f", "w", "#acacac"]

    fig = plt.figure()
    num_spectra = len(observed_spectra)

    axis = fig.add_subplot(num_spectra + 1, 1, 1)
    axis.axis('off')
    axis.set_title('$\chi^2$ = {chi_sq:.2f}, $N_{{dof}}$ = {num_dof:.0f}'
        .format(chi_sq=chi_sq, num_dof=num_dof))
 
    posteriors_formatted = "\n".join(["{parameter}:  {value:.3f}".format(parameter=parameter, value=posteriors[parameter])
        for parameter in sorted(posteriors.keys(), key=len)])

    axis.text(0.05, 0.05, posteriors_formatted, fontsize='x-small', transform=axis.transAxes)

    for i, (observed_spectrum, model_spectrum, mask) in enumerate(zip(observed_spectra, model_spectra, masks), start=2):
        axis = fig.add_subplot(num_spectra + 1, 1, i)

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
        
    return fig


def plot_all_results(results, output_prefix):
    """Plots and saves each result to file."""

    for i, result in enumerate(results, start=1):

        if result in (None, False): continue
        
        fig = plot_result(*result)
        if fig is None: continue

        filename = '{output_prefix}-star-{i}.png'.format(output_prefix=output_prefix, i=i)
        logging.info("Saved plotted result to {filename}".format(filename=filename))

        plt.savefig(filename)
        plt.close('all')

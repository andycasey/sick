# coding: utf-8

""" Handles the analysis for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import csv
import logging
import os
import multiprocessing
import pickle
import random
import sys
import time

from ast import literal_eval
from glob import glob

# Third-party
import emcee
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import models, utils, specutils

logger = logging.getLogger(__name__)



def initialise_priors(model, observations):
    """ Initialise the priors (or initial conditions) for the analysis """

    walker_priors = []
    initial_normalisation_coefficients = {}
    initial_normalisation_variances = {}

    nwalkers = model.configuration["solver"].get("nwalkers", 1)

    for i in xrange(nwalkers):

        current_walker = []
        for j, dimension in enumerate(model.dimensions):

            if dimension == "jitter":
                # Uniform prior between 0 and 1
                current_walker.append(random.uniform(0, 1))
                continue

            # Implicit priors
            if dimension.startswith("normalise_observed."):
                aperture = dimension.split(".")[1]
                coefficient_index = int(dimension.split(".")[-1].lstrip("a"))

                if aperture not in initial_normalisation_coefficients:
                    index = model._mapped_apertures.index(aperture)
                    order = model.configuration["normalise_observed"][aperture]["order"]

                    spectrum = observations[index]
                    
                    # Get the full range of spectra that will be normalised
                    if "masks" in model.configuration and aperture in model.configuration["masks"]:
                        
                        ranges = np.array(model.configuration["masks"][aperture])
                        min_range, max_range = np.min(ranges), np.max(ranges)
                        range_indices = np.searchsorted(spectrum.disp, [min_range, max_range])

                        flux_indices = np.zeros(len(spectrum.disp), dtype=bool)
                        flux_indices[range_indices[0]:range_indices[1]] = True
                        flux_indices *= np.isfinite(spectrum.flux) * (spectrum.flux > 0)
                        
                        logger.info("Normalising from {1:.0f} to {2:.0f} Angstroms in {0} aperture".format(
                            aperture, np.min(spectrum.disp[flux_indices]), np.max(spectrum.disp[flux_indices])))
                    else:
                        flux_indices = np.isfinite(spectrum.flux) * (spectrum.flux > 0) 

                    # Fit the spectrum with a polynomial of order X
                    coefficients, covariance = np.polyfit(spectrum.disp[flux_indices], spectrum.flux[flux_indices],
                        order, full=False, cov=True)

                    # Save the coefficients and variances
                    initial_normalisation_coefficients[aperture] = coefficients
                    initial_normalisation_variances[aperture] = np.diag(covariance)[::-1]

                coefficient = initial_normalisation_coefficients[aperture][coefficient_index]
                variance = initial_normalisation_variances[aperture][coefficient_index]
                
                spectrum = observations[model._mapped_apertures.index(aperture)]
                n = len(initial_normalisation_coefficients[aperture])

                sigma = 100./(np.mean(spectrum.disp[flux_indices])**(n - coefficient_index - 1))
                val = np.random.normal(coefficient, sigma)
                current_walker.append(val)
                print("coefficient", n -coefficient_index - 1, dimension, coefficient, val)
                print("original flux", np.mean(spectrum.flux[flux_indices]))

                #current_walker.append(np.random.normal(coefficient,
                #    0.1 * abs(coefficient) / np.mean(disp[flux_indices])**(coefficient_index)
                #    ))
                
                #current_walker.append(np.random.normal(coefficient, abs(sigma)))

                ##current_walker.append(np.random.normal(coefficient, \
                #    0.001 * abs(coefficient) / np.mean(observations[model._mapped_apertures.index(aperture)].disp)**coefficient_index))
                # This gives good setup in blue:
                #intrinsic_sigma = 1e-5
                #current_walker.append(random.normal(coefficient, intrinsic_sigma**float(coefficient_index + 1)))


                #current_walker.append(random.normal(coefficient, 0.01 * abs(coefficient)**(1.0/(coefficient_index + 1))))
                #current_walker.append(random.normal(coefficient, 0.1 * (abs(coefficient) / np.mean(observations[model._mapped_apertures.index(aperture)].disp)**(coefficient_index))))

                #if aperture == "blue":
                #    current_walker.append(random.normal(coefficient, 0.01 * abs(coefficient)))
                #else:
                #    current_walker.append(random.normal(coefficient, 0.01 * abs(coefficient)))
                #    #current_walker.append(random.normal(coefficient, 10**(-len(initial_normalisation_coefficients[aperture])) * abs(coefficient)))
                continue

            # Explicit priors
            prior_value = model.configuration["priors"][dimension]
            try:
                prior_value = float(prior_value)

            except:

                # We probably need to evaluate this.
                if prior_value.lower() == "uniform":
                    # Only works on stellar parameter values.
                    index = model.colnames.index(dimension)
                    possible_points = model.grid_points[:, index]

                    if i == 0: # Only print initialisation for the first walker
                        logging.info("Initialised {0} parameter with uniform distribution between {1:.2e} and {2:.2e}".format(
                            dimension, np.min(possible_points), np.max(possible_points)))
                    current_walker.append(random.uniform(np.min(possible_points), np.max(possible_points)))

                elif prior_value.lower().startswith("normal"):
                    mu, sigma = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                    if i == 0: # Only print initialisation for the first walker
                        logging.info("Initialised {0} parameter with a normal distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
                            dimension, mu, sigma))
                    current_walker.append(random.normal(mu, sigma))

                elif prior_value.lower().startswith("uniform"):
                    minimum, maximum = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                    if i == 0: # Only print initialisation for the first walker
                        logging.info("Initialised {0} parameter with a uniform distribution between {1:.2e} and {2:.2e}".format(
                            dimension, minimum, maximum))
                    current_walker.append(random.uniform(minimum, maximum))

                elif prior_value.lower().startswith("cross_correlate"):
                    # cross_correlate('data/sun.ms.fits', 8400, 8800)
                    raise NotImplementedError

                else:
                    raise TypeError("prior type not valid for {dimension}".format(dimension=dimension))

            else:
                if i == 0: # Only print initialisation for the first walker
                    logger_fn = logger.info if nwalkers == 1 else logger.warn
                    logger_fn("Initialised {0} parameter as a single value: {1:.2e}".format(
                        dimension, prior_value))

                current_walker.append(prior_value)

        # Add the walker
        if nwalkers == 1:
            walker_priors = current_walker
        
        else:
            walker_priors.append(current_walker)

    walker_priors = np.array(walker_priors)


    """
    print("PARAMETER NAMES", dimensions)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    red_indices = np.array([dimensions.index(name) for name in ["normalise_observed.red.a0","normalise_observed.red.a1","normalise_observed.red.a2","normalise_observed.red.a3"] if name in dimensions])

    fluxes = []
    for walker in walker_priors:
        coefficients = walker[red_indices]
        ax.plot(observations[0].disp, observations[0].flux / np.polyval(coefficients, observations[0].disp), c="#666666")

    #fluxes = np.array(fluxes)
    #ax.hist(fluxes[np.isfinite(fluxes)], bins=np.linspace(-5, 5, 50))
    
    fig.savefig("red-variance.png")
    

    fig = plt.figure()
    ax = fig.add_subplot(111)

    blue_indices = np.array([dimensions.index(name) for name in ["normalise_observed.blue.a0","normalise_observed.blue.a1","normalise_observed.blue.a2"]])
    fluxes = []
    for walker in walker_priors:
        coefficients = walker[blue_indices]
        ax.plot(observations[0].disp, observations[0].flux / np.polyval(coefficients, observations[0].disp), c="#666666")

    #fluxes = np.array(fluxes)
    #ax.hist(fluxes[np.isfinite(fluxes)], bins=np.linspace(-5, 5, 50))
    #ax.set_xlim(-5, 5)
    fig.savefig("blue-variance.png")


    plt.close("all")
    raise a
    """

    logging.info("Priors summary:")
    for i, dimension in enumerate(model.dimensions):
        if len(walker_priors.shape) > 1 and walker_priors.shape[1] > 1:
            logging.info("\tParameter {0} - mean: {1:.2e}, min: {2:.2e}, max: {3:.2e}".format(
                dimension, np.mean(walker_priors[:, i]), np.min(walker_priors[:, i]), np.max(walker_priors[:, i])))
        else:
            logging.info("\tParameter {0} - initial point: {1:.2e}".format(
                dimension, walker_priors[i]))

    return (model.dimensions, walker_priors)


def log_prior(theta, parameter_names, model):
    
    parameters = dict(zip(parameter_names, theta))

    for parameter, value in parameters.iteritems():
        # Check doppler shifts. Anything more than 500 km/s is considered implausible
        if parameter.startswith("doppler_shift.") and abs(value) > 500:
            return -np.inf

        # Check smoothing values. Any negative value is considered unrealistic
        if parameter.startswith("smooth_model_flux.") and 0 > value:
            return -np.inf

        # Check for jitter
        if parameter == "jitter" and not (1 > value > 0):
            return -np.inf

        # Check if point is within the grid boundaries?
        if parameter in model.grid_boundaries:
            min_value, max_value = model.grid_boundaries[parameter]
            if value > max_value or min_value > value:
                return -np.inf

    return 0

def log_likelihood(theta, parameter_names, model, observations, callback=None):
    """Calculates the likelihood that a given set of observations
    and parameters match the input models.

    parameters : list of `float`
        The free parameters to solve for. These are referenced in 
        `parameter_names`.

    observations : list of `Spectrum1D` objects
        The observed spectra.

    callback : function
        A callback to apply after completing the comparison.
    """

    blob = list(theta)
    if not np.isfinite(log_prior(theta, parameter_names, model)):
        logging.debug("Returning -inf log-likelihood because log-prior was -inf")
        return (-np.inf, blob + [-np.inf])

    parameters = dict(zip(parameter_names, theta))

    # Prepare the observed spectra: radial velocity shift? normalisation?
    observed_spectra = model.observed_spectra(observations, **parameters)
    if observed_spectra is None:
        logging.debug("Returning -inf log-likelihood because modified observed spectra is invalid")
        return (-np.inf, blob + [-np.inf])

    # Prepare the model spectra: smoothing? re-sample to observed dispersion?
    model_spectra = model.model_spectra(observations=observed_spectra, **parameters)
    if model_spectra is None:
        logging.debug("Returning -inf log-likelihood because modified model spectra is invalid")
        return (-np.inf, blob + [-np.inf])

    # Any masks?
    masks = model.masks(model_spectra)
    #weighting_functions = model.weights(model_spectra)
    
    # Calculate chi^2 difference
    chi_sqs = {}
    for i, (aperture, observed_spectrum) in enumerate(zip(model._mapped_apertures, observed_spectra)):


        inverse_variance = 1.0/(observed_spectrum.uncertainty**2 + parameters["jitter"])
        chi_sq = (observed_spectrum.flux - model_spectra[aperture].flux)**2 * inverse_variance
        
        # Apply any weighting functions to the chi_sq values
        #chi_sq /= weighting_functions[aperture](model_spectra[aperture].disp, model_spectra[aperture].flux)

        # Apply masks
        chi_sq *= masks[aperture]

        # Add only finite, positive values
        positive_finite_chisq_indices = (chi_sq > 0) * np.isfinite(chi_sq)
        positive_finite_flux_indices = (observed_spectrum.flux > 0) * np.isfinite(observed_spectrum.flux)

        # Useful_pixels of 1 indicates that we should use it, 0 indicates it was masked.
        useful_pixels = positive_finite_chisq_indices * positive_finite_flux_indices
        if sum(useful_pixels) == 0:
            logging.debug("Returning -np.inf log-likelihood because there were no useful pixels")
            return (-np.inf, blob + [-np.inf])

        chi_sqs[aperture] = np.sum(chi_sq[useful_pixels]) - np.sum(np.log(inverse_variance[useful_pixels]))

        # Update the masks values:
        #> -2: Not interested in this region, and it was non-finite (not used).
        #> -1: Interested in this region, but it was non-finite (not used).
        #>  0: Not interested in this region, it was finite (not used).
        #>  1: Interested in this region, it was finite (used for \chi^2 determination)
        masks[aperture][~useful_pixels] -= 2

    likelihood = -0.5 * np.sum(chi_sqs.values())

    logger.info("Returning log likelihood of {0:.2e} for parameters: {1}".format(likelihood,
        ", ".join(["{0} = {1:.2e}".format(name, value) for name, value in parameters.iteritems()])))  
    """
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0.10, bottom=0.10, top=0.95, right=0.95)

    ax = fig.add_subplot(211)
    ax.plot(model_spectra["blue"].disp/10., model_spectra["blue"].flux, 'b', zorder=10)
    ax.errorbar(observed_spectra[0].disp/10., observed_spectra[0].flux, fmt=None, ecolor="#666666",
        yerr=observed_spectra[0].uncertainty, zorder=-1)
    ax.plot(observed_spectra[0].disp/10., observed_spectra[0].flux, 'k', zorder=100)

    ax.set_xlim(450, 530)
    ax.set_ylim(0.6, 1.1)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalised Flux")

    logger.debug("a")
    fig = plt.figure(figsize=(10, 10))
    logger.debug("a1")
    fig.subplots_adjust(left=0.10, bottom=0.10, top=0.95, right=0.95)
    logger.debug("b")
    ax = fig.add_subplot(211)
    ax.plot(model_spectra["red"].disp/10., model_spectra["red"].flux, 'b', zorder=10)
    

    logger.debug("c", observed_spectra[0].disp, observed_spectra[0].flux, observed_spectra[0].uncertainty)
    ax.errorbar(observed_spectra[0].disp/10., observed_spectra[0].flux, fmt=None, ecolor="#666666",
        yerr=observed_spectra[0].uncertainty, zorder=-1)
    logger.debug("c1")
    ax.plot(observed_spectra[0].disp/10., observed_spectra[0].flux, 'k', zorder=100)


    logger.debug("d")
    ax.set_xlim(840, 885)
    ax.set_ylim(0.6, 1.1)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalised Flux")
    
    logger.debug("e")

    num = len(glob("progress-*.png"))
    
    logger.debug("f")
    plt.savefig("progress-{0}.png".format(num))
    
    logger.debug("g")
    plt.close("all")
    
    logger.debug("h")
    """
    return (likelihood, blob + [likelihood])


def solve(observed_spectra, model_filename, initial_guess=None):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    """

    # Check observed arms do not overlap
    observed_dispersions = [spectrum.disp for spectrum in observed_spectra]
    overlap = utils.find_spectral_overlap(observed_dispersions)
    if overlap is not None:
        raise ValueError("observed apertures cannot overlap in wavelength, but they do near {wavelength} Angstroms"
            .format(wavelength=overlap))

    # Load our model
    model = models.Model(model_filename)

    # Get the aperture mapping from observed spectra to model spectra
    # For example, which index in our list of spectra corresponds to
    # 'blue', or 'red' in our model 
    aperture_mapping = model.map_apertures(observed_dispersions)

    
    # Make fmin_powell the default
    if   model.configuration["solver"].get("method", "fmin_powell") == "fmin_powell":

        raise NotImplementedError
        parameter_names, p0 = initialise_priors(model, configuration, observed_spectra)
        parameters_final = scipy.optimize.fmin_powell(chi_sq, p0,
            args=(parameter_names, observed_spectra, model), xtol=0.001, ftol=0.001)


        return parameters_final

    elif model.configuration["solver"]["method"] == "emcee":

        # Ensure we have the number of walkers and steps specified in the configuration
        nwalkers, nsteps = model.configuration["solver"]["nwalkers"], \
            model.configuration["solver"]["burn"] + model.configuration["solver"]["sample"]

        lnprob0, rstate0 = None, None
        threads = model.configuration["solver"].get("threads", 1)

        mean_acceptance_fractions = np.zeros(nsteps)
        
        # Initialise priors and set up arguments for optimization
        parameter_names, p0 = initialise_priors(model, observed_spectra)

        logging.info("All priors initialsed for {0} walkers. Parameter names are: {1}".format(
            nwalkers, ", ".join(parameter_names)))

        # Initialise the sampler
        num_parameters = len(parameter_names)
        sampler = emcee.EnsembleSampler(nwalkers, num_parameters, log_likelihood,
            args=(parameter_names, model, observed_spectra),
            threads=threads)

        # BURN BABY BURN
        # Sample_data contains all the inputs, and the \chi^2 and L 
        # sampler_state = (pos, lnprob, state[, blobs])
        for i, sampler_state in enumerate(sampler.sample(
            p0, lnprob0=lnprob0, rstate0=rstate0, iterations=model.configuration["solver"]["burn"])):

            fraction_complete = (i + 1)/nsteps
            mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)

            # Announce progress
            logging.info("Sampler is {0:.2f}% complete (step {1:.0f}) with a mean acceptance fraction of {2:.3f}".format(
                fraction_complete * 100, i + 1, mean_acceptance_fractions[i]))

            if mean_acceptance_fractions[i] == 0:
                logging.warn("Mean acceptance fraction is zero. Breaking out of MCMC!")
                break

        sampler.reset()
        p0, lnpro0, state0 = sampler_state[0], None, None

        # SAMPLE ALL THE THINGS
        for j, sampler_state in enumerate(sampler.sample(
            p0, lnprob0=lnprob0, rstate0=rstate0, iterations=model.configuration["solver"]["sample"])):

            fraction_complete = (i + j + 1)/nsteps
            mean_acceptance_fractions[i + j] = np.mean(sampler.acceptance_fraction)

            # Announce progress
            logging.info("Sampler is {0:.2f}% complete (step {1:.0f}) with a mean acceptance fraction of {2:.3f}".format(
                fraction_complete * 100, i + j + 1, mean_acceptance_fractions[i + j]))

            if mean_acceptance_fractions[i + j] == 0:
                logging.warn("Mean acceptance fraction is zero. Breaking out of MCMC!")
                break


        # Convert state to posteriors
        logging.info("The final mean acceptance fraction is {0:.3f}".format(mean_acceptance_fractions[-1]))

        # Blobs contain all the sampled parameters and likelihoods        
        sampled = np.array(sampler.blobs).reshape((-1, num_parameters + 1))

        sampled = sampled[-int(model.configuration["solver"]["nwalkers"] * model.configuration["solver"]["sample"]):]
        sampled_theta, sampled_log_likelihood = sampled[:, :-1], sampled[:, -1]

        # Get the maximum estimate
        most_probable_index = np.argmax(sampled_log_likelihood)
        
        if not np.isfinite(sampled_log_likelihood[most_probable_index]):
            # You should probably check your configuration file for something peculiar
            raise ValueError("most probable sampled point was non-finite")
        
        # Get Maximum Likelihood values and their quantiles
        posteriors = {}
        for parameter_name, (me_value, pos_quantile, neg_quantile) in zip(parameter_names, 
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(sampled, [16, 50, 84], axis=0)))):
            posteriors[parameter_name] = (me_value, pos_quantile, neg_quantile)

        return (posteriors, sampler, model, mean_acceptance_fractions) 

            
    else:
        raise NotImplementedError("well well well, how did we find ourselves here, Mr Bond?")

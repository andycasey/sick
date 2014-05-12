# coding: utf-8

""" Analyse spectroscopic data """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["log_prior", "log_likelihood", "log_probability", "solve"]

# Standard library
import logging
import os
import multiprocessing
import re
from time import time

# Third-party
import acor
import emcee
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import models, priors, utils, specutils

logger = logging.getLogger(__name__.split(".")[0])

def log_prior(theta, model):
    """
    Return the prior for a set of theta given the model.

    Args:
        theta (list): The theta parameters that correspond with model.dimensions
        model (sick.models.Model object): The model class.
    
    Returns:
        The logarithmic prior for the parameters theta.
    """

    priors = model.configuration.get("priors", {})
    for dimension, value in zip(model.dimensions, theta):

        # Check smoothing values
        if dimension.startswith("convolve.") and 0 > value:
            return -np.inf

        # Check for jitter
        if dimension.startswith("jitter.") and not -10. < value < 1.:
            return -np.inf

        # Check if point is within the grid boundaries
        if dimension in model.grid_boundaries:
            if dimension not in priors:
                min_value, max_value = model.grid_boundaries[dimension]
                
            elif priors[dimension].startswith("uniform("):
                min_value, max_value = map(float, priors[dimension].split("(")[1].rstrip(")").split(","))

            if value > max_value or min_value > value:
                return -np.inf

    return 0


def log_likelihood(theta, model, observations):
    """
    Return the logarithmic likelihood for a set of theta given the model.

    Args:
        theta (list): The theta parameters that correspond with model.dimensions
        model (sick.models.Model object): The model class.
        observations (list of specutils.Spectrum1D objects): The data.
    """

    theta_dict = dict(zip(model.dimensions, theta))

    try:
        model_fluxes = model(observations=observations, **theta_dict)
    except:
        return -np.inf

    differences = []
    for (aperture, model_flux, observed_spectrum) in zip(model.apertures, model_fluxes, observations):

        inverse_variance = 1.0/(observed_spectrum.variance \
            + modelled_spectrum.flux**2 * np.exp(2. * theta_dict["jitter.{}".format(aperture)]))
        difference = (observed_spectrum.flux - model_flux)**2 * inverse_variance
        
        finite = np.isfinite(difference)
        differences.append(np.sum(difference[finite] - np.log(inverse_variance[finite])))

    likelihood = -0.5 * np.sum(differences)
    logger.debug("Returning log likelihood of {0:.2e} for parameters: {1}".format(likelihood,
        ", ".join(["{0} = {1:.2e}".format(name, value) for name, value in theta_dict.iteritems()])))  
   
    return likelihood


def log_probability(theta, model, observations):
    """
    Return the logarithmic probability (prior + likelihood) for theta given the data.

    Args:
        theta (list): The theta parameters that correspond with model.dimensions
        model (sick.models.Model object): The model class.
        observations (list of specutils.Spectrum1D objects): The data.
    """

    prior = log_prior(theta, model)
    if not np.isfinite(prior):
        logger.debug("Returning -inf log-likelihood because log-prior was -inf")
        return prior

    return prior + log_likelihood(theta, model, observations)


def sample_ball(point, observed_spectra, model):

    logger.info("Initialising priors around point {0}".format(point))

    # Create a sample ball around the result point
    ball_point = [point.get(dimension, 0) for dimension in model.dimensions]
    
    dimensional_std = []
    jitter_indices = []
    for di, dimension in enumerate(model.dimensions):

        if dimension in model.grid_points.dtype.names:
            dimensional_std.append(0.05 * np.ptp(model.grid_boundaries[dimension]))
           
        elif dimension.startswith("doppler_shift."):
            dimensional_std.append(5)
            
        elif dimension.startswith("smooth_model_flux."): 
            dimensional_std.append(0.10)

        elif dimension.startswith("normalise_observed."):

            if dimension.endswith(".s"): # spline smoothing
                s = np.random.normal(ball_point[di], np.sqrt(2*ball_point[di]))
                dimensional_std.append(np.max([s, 0]))

            else: #polynomial
                aperture = dimension.split(".")[1]
                coefficient = int(dimension.split(".")[2].lstrip("a"))
                order = model.configuration["normalise_observed"][aperture]["order"]
                observed_aperture = observed_spectra[model.apertures.index(aperture)]
                
                # This depends on the value of the polynomial coefficient, the dispersion, as well
                # as the order of the polynomial to fit the flux.

                # y = a*x^2 + b*x + c
                # y + delta_y = (a + delta_a)*x^2 + b*x + c
                # y + delta_y = a*x^2 + delta_a*x^2 + b*x + c
                # delta_y = delta_a*x^2
                # delta_a = <delta_y>/x^2

                # Since we want each coefficient to give an ~equal amount, we also divide by the
                # number of the coefficients:

                # delta_a = <delta_y>/(num_coefficients * x^2)
                # delta_b = <delta_y>/(num_coefficients * x^1)
                # delta_c = <delta_y>/(num_coefficients * x^0)

                # And we arbitrarily specify <delta_y> to be ~3x the mean uncertainty in flux.
                dispersion = observed_aperture.disp.mean()
                flux_scale = 3. * observed_aperture.uncertainty[np.isfinite(observed_aperture.uncertainty)].mean()
                dimensional_std.append(flux_scale/(dispersion**(order - coefficient)))
                   
        else:
            # Jitter
            dimensional_std.append(0.1)
            jitter_indices.append(di)
            
    walkers = model.configuration["solver"]["walkers"]
    p0 = emcee.utils.sample_ball(ball_point, dimensional_std, size=walkers)

    # Write over jitter priors
    for ji in jitter_indices:
        p0[:, ji] = np.random.uniform(-10, 1, size=walkers)

    # Write over normalisation priors if necessary
    for i, pi in enumerate(p0):

        # Model the flux, but don't normalise it.
        pi_parameters = dict(zip(model.dimensions, pi))
        for aperture in model.apertures:
            n = 0
            while "normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n) in pi_parameters.keys():
                pi_parameters["normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n)] = 0
                n += 1

            if n > 0:
                # Set the final coefficient as 1, so we end up having no normalisation
                pi_parameters["normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n-1)] = 1.

        if n > 0:
            try:
                model_apertures = model(observations=observed_spectra, **pi_parameters)
            except ValueError: continue
            
            for aperture, observed_aperture, model_aperture in zip(model.apertures, observed_spectra, model_apertures):

                continuum = (observed_aperture.flux + np.random.normal(0, observed_aperture.uncertainty))/model_aperture.flux
                finite = np.isfinite(continuum)

                # Get some normalisation coefficients
                coefficients = np.polyfit(model_aperture.disp[finite], continuum[finite], n-1)
                
                # Write over the prior values
                for j, coefficient in enumerate(coefficients):
                    index = model.dimensions.index("normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=j))
                    p0[i, index] = coefficient
    return p0


def __log_prob_of_implicit_theta(model, observed_spectra):
    theta = priors.prior(model, observed_spectra)
    return (theta, log_probability(theta, model, observed_spectra))

def __log_prob_of_explicit_theta(theta, model, observed_spectra):
    return (theta, log_probability(theta, model, observed_spectra))

def random_scattering(observed_spectra, model, initial_thetas=None):

    # Random scattering
    ta = time()
    threads = model.configuration["solver"].get("threads", 1)

    results = []
    def callback(result):
        results.append(result)

    if initial_thetas is None:

        initial_samples = model.configuration["solver"]["initial_samples"]

        if threads > 1:
            pool = multiprocessing.Pool(threads)
            for _ in xrange(initial_samples):
                pool.apply_async(__log_prob_of_implicit_theta, args=(model, observed_spectra), callback=callback)

            pool.close()
            pool.join()
        
        else:
            results = [__log_prob_of_initial_theta(model, observed_spectra) for _ in xrange(initial_samples)]

        logger.info("Calculating log probabilities of {0:.0f} implicit prior points took {1:.2f} seconds".format(
            initial_samples, time() - ta))

    else:

        if not isinstance(initial_thetas[0], (list, tuple, np.ndarray)):
            initial_thetas = [initial_thetas]

        if threads > 1:
            pool = multiprocessing.Pool(threads)

            for initial_theta in initial_thetas:
                pool.apply_async(__log_prob_of_explicit_theta, args=(initial_theta, model, observed_spectra), callback=callback)

            pool.close()
            pool.join()

        else:
            results = [__log_prob_of_explicit_theta(initial_theta, model, observed_spectra) for initial_theta in initial_thetas]

        logger.info("Calculating log probabilities of {0:.0f} implicit prior points took {1:.2f} seconds".format(
            len(initial_thetas), time() - ta))

    index = np.argmax([result[1] for result in results])
    p0 = results[index][0]
    return p0


def solve(observed_spectra, model, initial_thetas=None):
    """
    Solve for the model parameters theta given the observed spectra.

    Args:
        observed_spectra (list of specutils.Spectrum1D objects): The observed spectra.
        model (sick.models.Model object): The model class.
    """

    t_init = time()

    # Load our model if necessary
    if not isinstance(model, models.Model):
        model = models.Model(model)

    # Set the aperture mapping from observed spectra to model spectra
    # For example, which index in our list of spectra corresponds to
    # 'blue', or 'red' in our model
    model.map_channels(observed_spectra)
    
    # Perform any optimisation and initialise priors
    if model.configuration["solver"].get("optimise", True):
        
        p0 = random_scattering(observed_spectra, model, initial_thetas)

        logger.info("Optimising from point:")
        for dimension, value in zip(model.dimensions, p0):
            logger.info("\t{0}: {1:.2e}".format(dimension, value))

        ta = time()

        # Optimisation
        opt_theta, fopt, niter, funcalls, warnflag = scipy.optimize.fmin(
            lambda theta, model, obs: -log_probability(theta, model, obs), p0,
            args=(model, observed_spectra), xtol=0.001, ftol=0.001, full_output=True, disp=False)

        if warnflag > 0:
            messages = [
                "Maximum number of function evaluations made. Optimised solution may be inaccurate.",
                "Maximum number of iterations reached. Optimised solution may be inaccurate."
            ]
            logger.warn(messages[warnflag - 1])
        logger.info("Optimisation took {0:.2f} seconds".format(time() - ta))

        # Sample around opt_theta using some sensible things
        p0 = sample_ball(dict(zip(model.dimensions, opt_theta)), observed_spectra, model)

    else:
        warnflag, p0 = 0, priors.explicit(model, observed_spectra)

    logger.info("Priors summary:")
    for i, dimension in enumerate(model.dimensions):
        if len(p0.shape) > 1 and p0.shape[1] > 1:
            logger.info("\tParameter {0} - mean: {1:.4e}, std: {2:.4e}, min: {3:.4e}, max: {4:.4e}".format(
                dimension, np.mean(p0[:, i]), np.std(p0[:, i]), np.min(p0[:, i]), np.max(p0[:, i])))
        else:
            logger.info("\tParameter {0} - initial point: {1:.2e}".format(dimension, p0[i]))

    # Initialise the sampler
    walkers, burn, sample = [model.configuration["solver"][k] for k in ("walkers", "burn", "sample")]
    mean_acceptance_fractions = np.zeros(burn)
    autocorrelation_time = np.zeros((burn, len(model.dimensions)))

    sampler = emcee.EnsembleSampler(walkers, len(model.dimensions), log_probability,
        args=(model, observed_spectra), threads=model.configuration["solver"].get("threads", 1))

    # Start sampling
    for i, (pos, lnprob, rstate) in enumerate(sampler.sample(p0, iterations=burn)):
        mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)
        
        # Announce progress
        logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f}, maximum log likelihood"
            " in last step was {2:.3e}".format(i + 1, mean_acceptance_fractions[i],
                np.max(sampler.lnprobability[:, i])))

        if mean_acceptance_fractions[i] in (0, 1):
            raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(mean_acceptance_fractions[i]))

    chain, ln_probability = sampler.chain, sampler.lnprobability

    logger.info("Resetting chain...")
    sampler.reset()

    logger.info("Sampling 'posterior'?")
    sampler.run_mcmc(pos, model.configuration["solver"]["sample"], rstate0=rstate)

    # Get the quantiles
    posteriors = {}
    for parameter_name, (median, quantile_16, quantile_84) in zip(model.dimensions, 
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(sampler.chain.reshape(-1, len(model.dimensions)), [16, 50, 84], axis=0)))):
        posteriors[parameter_name] = (median, quantile_16, quantile_84)

    # Convert posteriors and chains to appropriate units
    # TODO

    if not (0.5 > mean_acceptance_fractions[-1] > 0.2):
        warnflag += 4

    # Send back additional information
    additional_info = {
        "priors": p0,
        "chain": chain,
        "ln_probability": ln_probability,
        "mean_acceptance_fractions": mean_acceptance_fractions,
        "warnflag": warnflag,
        "time_elapsed": time() - t_init
    }

    logger.info("Completed in {0:.2f} seconds".format(additional_info["time_elapsed"]))

    return (posteriors, sampler, additional_info)


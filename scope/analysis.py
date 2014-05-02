# coding: utf-8

""" Handles the analysis for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["initialise_priors", "log_likelihood", "solve"]

# Standard library
import logging
import os
import multiprocessing
from time import time

# Third-party
import acor
import emcee
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import models, utils, specutils

logger = logging.getLogger(__name__.split(".")[0])


def implicit_prior(model, observations, size=1):
    """Generate implicit priors for all dimensions in the given model"""

    priors = []
    while size > len(priors):

        walker_prior = []
        normalisation_coefficients = {}

        for i, dimension in enumerate(model.dimensions):

            # Is this a grid dimension?
            # If it is, let's assume it's uniformly distributed
            if dimension in model.grid_points.dtype.names:
                walker_prior.append(random.uniform(*model.grid_boundaries[dimension]))
                continue

            # The grid dimensions come first in model.dimensions, so once the length
            # of walker_prior matches model.grid_points.dtype.names, we have what we
            # need to interpolate a flux. The interpolated flux will can be used for
            # the other priors
            if len(model.grid_points.dtype.names) == len(walker_prior):
                # Attempt to interpolate a flux
                interpolated_fluxes = model.interpolate_flux(walker_prior)

                non_finite = lambda _: np.all(~np.isfinite(_))
                if all(map(non_finite, interpolated_fluxes.values())): 
                    break

            # Velocities
            if dimension.startswith("doppler_shift."):
                # Assumes a velocity, not a redshift
                #velocity = random.normal(0, 100)
                #walker_prior.append(velocity)

                aperture = dimension.split(".")[1]
                observed_aperture = observations[model._mapped_apertures.index(aperture)]
                model_aperture = specutils.Spectrum1D(disp=model.dispersion[aperture], flux=interpolated_fluxes[aperture])
                try:
                    v_rad, u_v_rad, r = observed_aperture.cross_correlate(model_aperture)

                except (ValueError, ):
                    walker_prior.append(random.normal(0, 100))
                    
                else:
                    if abs(v_rad) > 500:
                        walker_prior.append(random.normal(0, 100))

                    else:
                        walker_prior.append(v_rad)

            # Smoothing
            elif dimension.startswith("smooth_model_flux."):

                aperture = dimension.split(".")[1]

                walker_prior.append({"blue": 1.8, "red": 0.8}[aperture])

                """
                # Estimate this by the observed spectral resolution
                observed_aperture = observations[model._mapped_apertures.index(aperture)]
                observed_spectral_resolution = observed_aperture.disp[1:]/np.diff(observed_aperture.disp)

                model_spectral_resolution = model.dispersion[aperture][1:]/np.diff(model.dispersion[aperture])

                resolution_difference = np.mean(model_spectral_resolution)/np.mean(observed_spectral_resolution)

                resolution_difference /= (2 * (2 * np.log(2))**0.5)

                walker_prior.append(random.normal(resolution_difference, 0.1 * resolution_difference))
                """

            # Normalisation
            elif dimension.startswith("normalise_observed."):

                # We will use the interpolated flux to determine approximate normalisation
                # coefficients
                aperture = dimension.split(".")[1]
                if aperture not in normalisation_coefficients.keys():

                    if not model.configuration["normalise_observed"][aperture]["perform"] \
                    or model.configuration["normalise_observed"][aperture]["method"] == "spline": continue

                    model_flux = interpolated_fluxes[aperture]

                    order = model.configuration["normalise_observed"][aperture]["order"]
                    aperture_index = model._mapped_apertures.index(aperture)
                    observed_aperture = observations[aperture_index]

                    # Get masks
                    # TODO

                    interpolated_model_flux = np.interp(observed_aperture.disp,
                        model.dispersion[aperture], model_flux, left=np.nan, right=np.nan)
                    continuum = observed_aperture.flux/interpolated_model_flux

                    finite = np.isfinite(continuum)
                    coefficients = np.polyfit(observed_aperture.disp[finite], continuum[finite], order)
                    
                    normalisation_coefficients[aperture] = coefficients

                coefficient = int(dimension.split(".")[2].lstrip("a"))
                walker_prior.append(normalisation_coefficients[aperture][coefficient])

            # Jitter
            elif dimension == "jitter" or dimension.startswith("jitter."):
                # Uniform between -10 and 1
                walker_prior.append(random.uniform(-10, 1))

            else:
                raise RuntimeError("don't know how to generate implicit priors from dimension {0}".format(dimension))

        else:
            priors.append(walker_prior)

    priors = np.array(priors)
    if size == 1:
        priors = priors.flatten()

    return priors






def initialise_priors(model, observations):
    """ Initialise the priors (or initial conditions) for the analysis """

    walker_priors = []
    initial_normalisation_coefficients = {}

    walkers = model.configuration["solver"].get("walkers", 1)

    if model.configuration.has_key("priors"):
        while walkers > len(walker_priors):

            current_walker = []
            interpolated_flux = {}
            initial_normalisation_coefficients = {}
            for j, dimension in enumerate(model.dimensions):

                # Have we just finished doing the model dimensions?
                # If so then we can interpolate to a model flux
                if len(current_walker) == len(model.grid_points.dtype.names):
                    interpolated_flux = model.interpolate_flux(current_walker)

                    if np.all(~np.isfinite(interpolated_flux.values()[0])):
                        # None of the flux in the first beam are finite.

                        break

                # Jitter
                if dimension == "jitter" or dimension.startswith("jitter."):
                    # Uniform prior between 0 and 1
                    current_walker.append(random.uniform(-10, 1))
                    continue

                # Implicit priors
                if dimension.startswith("normalise_observed."):

                    aperture = dimension.split(".")[1]
                    coefficient_index = int(dimension.split(".")[-1].lstrip("a"))

                    # If we're at this stage we should have grid point dimensions
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
                            flux_indices *= np.isfinite(spectrum.flux)
                            
                            logger.debug("Normalising from {1:.0f} to {2:.0f} Angstroms in {0} aperture".format(
                                aperture, np.min(spectrum.disp[flux_indices]), np.max(spectrum.disp[flux_indices])))
                        else:
                            flux_indices = np.isfinite(spectrum.flux)

                        # Fit the spectrum with a polynomial of order X
                        resampled_interpolated_flux = np.interp(spectrum.disp[flux_indices], model.dispersion[aperture],
                            interpolated_flux[aperture])
                        coefficients = np.polyfit(spectrum.disp[flux_indices], spectrum.flux[flux_indices]/resampled_interpolated_flux, order)
                        
                        # Save the coefficients and variances
                        initial_normalisation_coefficients[aperture] = coefficients
                    
                    coefficient = initial_normalisation_coefficients[aperture][coefficient_index]
                    current_walker.append(coefficient)

                    continue

                # Explicit priors
                prior_value = model.configuration["priors"][dimension]
                try:
                    prior_value = float(prior_value)

                except:

                    # We probably need to evaluate this.
                    if prior_value.lower() == "uniform":
                        # Only works on stellar parameter values.
                        possible_points = model.grid_points[dimension].view(np.float)

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with uniform distribution between {1:.2e} and {2:.2e}".format(
                                dimension, np.min(possible_points), np.max(possible_points)))
                        current_walker.append(random.uniform(np.min(possible_points), np.max(possible_points)))

                    elif prior_value.lower().startswith("normal"):
                        mu, sigma = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with a normal distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
                                dimension, mu, sigma))
                        current_walker.append(random.normal(mu, sigma))

                    elif prior_value.lower().startswith("uniform"):
                        minimum, maximum = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with a uniform distribution between {1:.2e} and {2:.2e}".format(
                                dimension, minimum, maximum))
                        current_walker.append(random.uniform(minimum, maximum))

                    elif prior_value.lower() == "cross_correlate":
                        current_walker.append(random.normal(0, 100))

                        #aperture = dimension.split(".")[1]
                        #observed_spectrum = observations[model._mapped_apertures.index(aperture)]

                        #model_aperture_spectrum = specutils.Spectrum1D(model.dispersion[aperture], interpolated_flux[aperture])
                        #v_rad, u_v_rad, R = observed_spectrum.cross_correlate(model_aperture_spectrum)
                        #current_walker.append(random.normal(v_rad, u_v_rad))

                    else:
                        raise TypeError("prior type not valid for {dimension}".format(dimension=dimension))

                else:
                    if len(walker_priors) == 0: # Only print initialisation for the first walker
                        logger_fn = logger.info if walkers == 1 else logger.warn
                        logger_fn("Initialised {0} parameter as a single value: {1:.2e}".format(dimension, prior_value))

                    current_walker.append(prior_value)

            
            # Was the walker actually valid?
            if len(current_walker) > 0:

                # Add the walker
                if walkers == 1:
                    walker_priors = current_walker
                
                else:
                    walker_priors.append(current_walker)

        walker_priors = np.array(walker_priors)

    else:
        walker_priors = implicit_prior(model, observations, size=walkers)

    return walker_priors


def log_prior(theta, model):
    
    for parameter, value in zip(model.dimensions, theta):
        # Check smoothing values
        if parameter.startswith("smooth_model_flux.") and not 10 > value >= 0:
            return -np.inf

        # Check for jitter
        if (parameter == "jitter" or parameter.startswith("jitter.")) \
        and not -10. < value < 1.:
            return -np.inf

        # Check if point is within the grid boundaries
        if parameter in model.grid_boundaries:
            min_value, max_value = model.grid_boundaries[parameter]
            if value > max_value or min_value > value:
                return -np.inf

    return 0


def log_likelihood(theta, model, observations):
    """Calculates the likelihood that a given set of observations
    and parameters match the input models.

    parameters : list of `float`s
        The free parameters to solve for. These are referenced in 
        `parameter_names`.

    observations : list of `Spectrum1D` objects
        The observed spectra.

    callback : function
        A callback to apply after completing the comparison.
    """


    parameters = dict(zip(model.dimensions, theta))

    # Prepare the model spectra
    model_spectra = model.model_spectra(observations=observations, **parameters)

    if model_spectra is None:
        logger.debug("Returning -inf log-likelihood because model spectra is non-finite")
        return -np.inf

    chi_sqs = []
    masks = model.masks(parameters, model_spectra)
    for i, (aperture, modelled_spectrum, observed_spectrum) in enumerate(zip(model._mapped_apertures, model_spectra, observations)):

        inverse_variance = 1.0/(observed_spectrum.uncertainty**2 + modelled_spectrum.flux**2 * np.exp(2. * parameters["jitter.{0}".format(aperture)]))
        chi_sq = (observed_spectrum.flux - modelled_spectrum.flux)**2 * inverse_variance

        # Any (rest-frame) masks to apply?
        chi_sq *= masks[aperture]
        
        useful_pixels = np.isfinite(chi_sq) 
        chi_sqs.append(np.sum(chi_sq[useful_pixels] - np.log(inverse_variance[useful_pixels])))

    likelihood = -0.5 * np.sum(chi_sqs)

    logger.debug("Returning log likelihood of {0:.2e} for parameters: {1}".format(likelihood,
        ", ".join(["{0} = {1:.2e}".format(name, value) for name, value in parameters.iteritems()])))  
   
    return  likelihood


def log_probability(theta, model, observations):

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

            aperture = dimension.split(".")[1]
            coefficient = int(dimension.split(".")[2].lstrip("a"))
            order = model.configuration["normalise_observed"][aperture]["order"]
            observed_aperture = observed_spectra[model._mapped_apertures.index(aperture)]
            
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

    # Write over normalisation priors

    for i, pi in enumerate(p0):

        # Model the flux, but don't normalise it.
        pi_parameters = dict(zip(model.dimensions, pi))
        for aperture in model.apertures:
            n = 0
            while "normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n) in pi_parameters.keys():
                pi_parameters["normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n)] = 0
                n += 1

            # Set the final coefficient as 1, so we end up having no normalisation
            pi_parameters["normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n-1)] = 1.

        model_apertures = model.model_spectra(observations=observed_spectra, **pi_parameters)
        if model_apertures is None: continue

        if n > 0:
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


def __log_prob_of_implicit_prior(model, observed_spectra):
    theta = implicit_prior(model, observed_spectra)
    ln_prob = log_probability(theta, model, observed_spectra)
    return (theta, ln_prob)


def solve(observed_spectra, model):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    """

    t_init = time()

    # Load our model if necessary
    if not isinstance(model, models.Model):
        model = models.Model(model)

    # Set the aperture mapping from observed spectra to model spectra
    # For example, which index in our list of spectra corresponds to
    # 'blue', or 'red' in our model
    model.map_apertures(observed_spectra)
    
    # Perform any optimisation and initialise priors
    if model.configuration["solver"].get("optimise", True):
        
        # Random scattering
        ta = time()
        pool = multiprocessing.Pool(model.configuration["solver"].get("threads", 1))
        initial_samples = model.configuration["solver"].get("initial_samples", 1000)

        results = []
        def callback(result):
            results.append(result)

        for i in xrange(initial_samples):
            pool.apply_async(__log_prob_of_implicit_prior, args=(model, observed_spectra), callback=callback)

        pool.close()
        pool.join()
        
        index = np.argmax([result[1] for result in results])
        p0 = results[index][0]

        logger.info("Calculating log probabilities of {0:.0f} implicit prior points took {1:.2f} seconds".format(
            initial_samples, time() - ta))

        logger.info("Optimising from point with log probability {0:.3e}:".format(results[index][1]))
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
        warnflag, p0 = 0, initialise_priors(model, observed_spectra)

    logger.info("Priors summary:")
    for i, dimension in enumerate(model.dimensions):
        if len(p0.shape) > 1 and p0.shape[1] > 1:
            logger.info("\tParameter {0} - mean: {1:.4e}, std: {2:.4e}, min: {3:.4e}, max: {4:.4e}".format(
                dimension, np.mean(p0[:, i]), np.std(p0[:, i]), np.min(p0[:, i]), np.max(p0[:, i])))
        else:
            logger.info("\tParameter {0} - initial point: {1:.2e}".format(dimension, p0[i]))

    # Get the number of walkers, etc
    walkers, burn, sample = [model.configuration["solver"][k] for k in ("walkers", "burn", "sample")]
    mean_acceptance_fractions = np.zeros(burn)
    autocorrelation_time = np.zeros((burn, len(model.dimensions)))

    # Initialise the sampler
    sampler = emcee.EnsembleSampler(walkers, len(model.dimensions), log_probability,
        args=(model, observed_spectra), threads=model.configuration["solver"].get("threads", 1))

    # Start sampling
    for i, (pos, lnprob, rstate) in enumerate(sampler.sample(p0, iterations=burn)):
        mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)
        
        # Announce progress
        logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f}, maximum log likelihood in last step was {2:.3e}".format(
            i + 1, mean_acceptance_fractions[i], np.max(sampler.lnprobability[:, i])))

        if mean_acceptance_fractions[i] in (0, 1):
            raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(mean_acceptance_fractions[i]))

    chain, ln_probability = sampler.chain, sampler.lnprobability

    logger.info("Resetting chain...")
    sampler.reset()
    logger.info("Sampling 'posterior'?")
    sampler.run_mcmc(pos, model.configuration["solver"].get("sample", 100))

    # Get the quantiles
    posteriors = {}
    for parameter_name, (median, quantile_16, quantile_84) in zip(model.dimensions, 
        map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(sampler.chain.reshape(-1, len(model.dimensions)), [16, 50, 84], axis=0)))):
        posteriors[parameter_name] = (median, quantile_16, quantile_84)

    if mean_acceptance_fractions[-1] < 0.25:
        warnflag += 3

    # Send back additional information
    additional_info = {
        "priors": p0,
        "chain": chain,
        "ln_probability": ln_probability,
        "mean_acceptance_fractions": mean_acceptance_fractions,
        "warnflag": warnflag
    }

    logger.info("Completed in {0:.2f} seconds".format(time() - t_init))

    return (posteriors, sampler, additional_info)


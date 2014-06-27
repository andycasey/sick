# coding: utf-8

""" Analyse spectroscopic data """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["initial_point", "log_prior", "log_likelihood", "log_probability", "solve"]

# Standard library
import logging
import multiprocessing.pool
import os
import threading
import re
from functools import partial
from itertools import chain
from time import time

# Third-party
import acor
import emcee
import numpy as np
import numpy.random as random
from scipy import optimize, stats, ndimage

# Module
import models, utils, specutils

logger = logging.getLogger(__name__.split(".")[0])

_prior_eval_env_ = { 
    "locals": None, "globals": None, "__name__": None, "__file__": None, "__builtins__": None,
    "uniform": lambda a, b: partial(stats.uniform.logpdf, **{"loc": a, "scale": b - a}),
    "normal": lambda a, b: partial(stats.norm.logpdf, **{"loc": a, "scale": b})
}

def initial_point(evaluated_priors, model, observations):
    """
    Return an initial theta point for the model, given the data.

    Args:
        model (sick.models.Model object): The model class.
        observations (list of specutils.Spectrum1D objects): The observed spectra.
    """

    # Set the default priors:
    # Default doppler shift priors
    initial_distributions = {}
    initial_distributions.update(dict(
        [("z.{0}".format(c), "cross_correlate({0})".format(c)) for c in model.channels]
    ))

    # Default smoothing priors should be absolute normals
    initial_distributions.update(dict(zip(
        ["convolve.{0}".format(channel) for channel in model.channels],
        ["abs(normal(0, 1))"] * len(model.channels)
    )))

    # Default outlier distribution priors
    initial_distributions.update({"Pb": "uniform(0, 1)"})
    all_fluxes = np.array(list(chain(*[each.flux for each in observations])))
    all_fluxes = all_fluxes[np.isfinite(all_fluxes)]

    initial_distributions.update({
        "Yb": "normal({0}, 0.5 * {0})".format(np.median(all_fluxes)),
        "Vb": "normal({0}, 0.5 * {0})".format(np.std(all_fluxes)**2)
        })

    # Environment variables for explicit priors
    # The channel names will be passed to contain all the information required
    # for cross-correlation
    env = { "locals": None, "globals": None, "__name__": None, "__file__": None,
        "__builtins__": None, "normal": random.normal, "uniform": random.uniform,
        "cross_correlate": specutils.__cross_correlate__, "abs": abs, }

    # Overwrite initial_distributions with any explicit priors:
    initial_distributions.update(model.priors.copy())

    current_point = []
    model_intensities = {}
    continuum_parameters = {}

    if evaluated_priors is None:
        evaluated_priors = {}
    else:
        evaluated_priors = evaluated_priors.copy()

    scaled_observations = {}
    
    for i, dimension in enumerate(model.dimensions):

        # Have we already evaluated this dimension?
        if dimension in evaluated_priors.keys():
            current_point.append(evaluated_priors[dimension])
            continue

        # Have we got priors for all grid points?
        if len(current_point) == len(model.grid_points.dtype.names):

            # Interpolate intensities at this point
            try:
                model_intensities.update(model.interpolate_flux(current_point))
            except (IndexError, ValueError) as e:
                break

            # Check the intensities are finite, otherwise move on
            if not np.all(map(np.all, map(np.isfinite, model_intensities.values()))):
                break

            # Smooth the model intensities if required
            for channel in model_intensities.keys():
                
                if "normalise.{0}.bandwidth".format(channel) in model.dimensions: 
                    evaluated_priors["normalise.{0}.bandwidth".format(channel)] = np.random.normal(1000, 1000)
                if "normalise.{0}.s_scale".format(channel) in model.dimensions:
                    evaluated_priors["normalise.{0}.s_scale".format(channel)] = np.random.normal(1, 0.1)

                if "convolve.{0}".format(channel) in model.dimensions:

                    # We have to evaluate this prior now
                    sigma = eval(initial_distributions["convolve.{0}".format(channel)], env)
                    kernel = (sigma/(2 * (2*np.log(2))**0.5))/np.mean(np.diff(model.dispersion[channel]))
                    
                    evaluated_priors["convolve.{0}".format(channel)] = sigma

                    # Smooth the model intensities
                    model_intensities[channel] = ndimage.gaussian_filter1d(model_intensities[channel], kernel)

                # Get continuum knots/coefficients for each aperture
                observed_channel = observations[model.channels.index(channel)]

                fft_fitler = 1.
                if "normalise.{0}.bandwidth".format(channel) in evaluated_priors:
                    fft_fitler = observed_channel.fft(evaluated_priors["normalise.{0}.bandwidth".format(channel)])

                continuum = (observed_channel.flux / fft_fitler)/np.interp(observed_channel.disp,
                    model.dispersion[channel], model_intensities[channel], left=np.nan,
                    right=np.nan)

                finite_continuum = np.isfinite(continuum)

                if model.configuration.get("normalise", False) and channel in model.configuration["normalise"]:
                    method = model.configuration["normalise"][channel]["method"]

                else:
                    method = None

                # Re-interpolate the observed fluxes where they are nans
                finite_observed_fluxes = np.isfinite(observed_channel.flux)
                cleaned_observed_flux = np.interp(observed_channel.disp,
                    observed_channel.disp[finite_observed_fluxes], observed_channel.flux[finite_observed_fluxes])

                # Re-bin onto log-lambda scale
                log_delta = np.diff(observed_channel.disp).min()
                wl_min, wl_max = observed_channel.disp.min(), observed_channel.disp.max()
                log_observed_dispersion = np.exp(np.arange(np.log(wl_min), np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

                # Scale the intensities to the data
                interpolated_model_intensities = np.interp(log_observed_dispersion, model.dispersion[channel],
                    model_intensities[channel], left=np.nan, right=np.nan)

                # Get only finite overlap
                finite = np.isfinite(interpolated_model_intensities)

                if method == "polynomial":
                    # Fit polynomial coefficients


                    order = model.configuration["normalise"][channel]["order"]
                    continuum_parameters[channel] = np.polyfit(observed_channel.disp[finite_continuum], continuum[finite_continuum], order)

                    # Transform the observed data
                    observed_scaled_intensities = cleaned_observed_flux \
                        / np.polyval(continuum_parameters[channel], observed_channel.disp)
                    interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                        observed_scaled_intensities, left=1, right=1)

                    env.update({channel:
                        (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])
                    })

                elif method == "spline":
                    num_knots = model.configuration["normalise"][channel]["knots"]

                    # Determine knot spacing
                    finite_continuum = np.isfinite(continuum)
                    knot_spacing = np.ptp(observed_channel.disp[finite_continuum])/(num_knots + 1)
                    continuum_parameters[channel] = np.arange(observed_channel.disp[finite_continuum][0] + knot_spacing,
                        observed_channel.disp[finite_continuum][-1] + knot_spacing, knot_spacing)[:num_knots]

                    tck = interpolate.splrep(observed_channel.disp[finite_continuum], continuum[finite_continuum],
                        w=1./np.sqrt(observed_channel.variance[finite_continuum]), t=continuum_parameters[channel])

                    # Transform the observed data
                    observed_scaled_intensities = cleaned_observed_flux \
                        / interpolate.splev(observed_channel.disp, tck)
                    interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                        observed_scaled_intensities, left=1, right=1)

                    env.update({channel:
                        (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])
                    })

                else:
                    # No normalisation specified, but we might be required to get a cross-correlation prior.
                    interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                        cleaned_observed_flux, left=1, right=1)
                    env.update({channel:
                        (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])})

        # Is there an explicitly specified distribution for this dimension?
        specified_prior = initial_distributions.get(dimension, False)
        if specified_prior:
            # Evaluate the prior given the environment information
            current_point.append(eval(specified_prior, env))
            continue

        # These are all implicit priors from here onwards.
        if dimension.startswith("normalise."):

            if dimension.endswith(".bandwidth") or dimension.endswith(".s_scale"): continue

            # Get the coefficient
            channel = dimension.split(".")[1]
            coefficient_index = int(dimension.split(".")[2][1:])
            coefficient_value = continuum_parameters[channel][coefficient_index]

            # Polynomial coefficients will introduce their own scatter
            # if the method is a spline, we should produce some scatter around the points
            method = model.configuration["normalise"][channel]["method"]
            if method == "polynomial":
                current_point.append(coefficient_value)

            elif method == "spline":
                # Get the difference between knot points
                knot_sigma = np.mean(np.diff(continuum_parameters[channel]))/10.
                current_point.append(random.normal(coefficient_value, knot_sigma))

        else:
            raise KeyError("Cannot interpret initial scattering distribution for {0}".format(dimension))

    # Check that we have the full number of walker values
    if len(current_point) == len(model.dimensions):
        return current_point

    else:
        return current_point + [0.] * (len(model.dimensions) - len(current_point))


def log_prior(theta, model):
    """
    Return the prior for a set of theta given the model.

    Args:
        theta (list): The theta parameters that correspond with model.dimensions
        model (sick.models.Model object): The model class.
    
    Returns:
        The logarithmic prior for the parameters theta.
    """

    log_prior = 0
    for dimension, value in zip(model.dimensions, theta):

        if dimension in model.priors:
            f = eval(model.priors[dimension], _prior_eval_env_)
            log_prior += f(value)

        # Check smoothing values
        if dimension.startswith("convolve.") and 0 > value:
            return -np.inf

        # Check for outlier parameters
        if dimension == "Pb" and not (1. > value > 0.) \
        or dimension == "Vb" and 0 > value:
            return -np.inf

        # Check for fourier filter parameters
        if dimension.startswith("normalise."):
            if dimension.endswith(".bandwidth") and 0 >= value:
                return -np.inf
            if dimension.endswith(".s_scale") and 0 >= value:
                return -np.inf

    logging.debug("Returning log-prior of {0:.2e} for parameters: {1}".format(log_prior,
        ", ".join(["{0} = {1:.2e}".format(name, value) for name, value in zip(model.dimensions, theta)])))
    return log_prior


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
        model_fluxes, model_continua = model(observations=observations, full_output=True, **theta_dict)
    except:
        return -np.inf

    likelihood = 0
    for (channel, model_flux, model_continuum, observed_spectrum) in zip(model.channels, model_fluxes, model_continua, observations):

        signal_inverse_variance = 1.0/(observed_spectrum.variance \
            + model_flux**2 * np.exp(2. * theta_dict["jitter.{0}".format(channel)]))

        signal_likelihood = -0.5 * ((observed_spectrum.flux - model_flux)**2 * signal_inverse_variance \
            - np.log(signal_inverse_variance))

        # Are we modelling the outliers as well?
        if "Pb" in theta_dict.keys():
            outlier_inverse_variance = 1.0/(theta_dict["Vb"] + observed_spectrum.variance \
                + model_flux**2 * np.exp(2. * theta_dict["jitter.{0}".format(channel)]))

            #continuum = model._continuum(channel, observed_spectrum, model_flux, **theta_dict)
            outlier_likelihood = -0.5 * ((observed_spectrum.flux - model_continuum)**2 * outlier_inverse_variance \
                - np.log(outlier_inverse_variance))

            Pb = theta_dict["Pb"]
            finite = np.isfinite(outlier_likelihood * signal_likelihood)
            likelihood += np.sum(np.logaddexp(np.log(1. - Pb) + signal_likelihood[finite], np.log(Pb) + outlier_likelihood[finite]))

        else:
            finite = np.isfinite(signal_likelihood)
            likelihood += np.sum(signal_likelihood[finite])
    if likelihood == 0:
        return -np.inf

    logger.debug("Returning log-likelihood of {0:.2e} for parameters: {1}".format(likelihood,
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
    if np.isinf(prior):
        logger.debug("Returning -inf log-probability because log-prior was -inf")
        return prior

    return prior + log_likelihood(theta, model, observations)


def sample_ball(point, observed_spectra, model):
    """
    Return a multi-dimensional Gaussian around a ball point.

    """ 
    
    # Create a sample ball around the result point
    ball_point = [point.get(dimension, 0) for dimension in model.dimensions]
    
    dimensional_std = []
    jitter_indices = []
    for di, dimension in enumerate(model.dimensions):

        if dimension in model.grid_points.dtype.names:
            # Set sigma to be 30% of the dimension dynamic range
            dimensional_std.append(0.05 * np.ptp(model.grid_boundaries[dimension]))
           
        elif dimension.startswith("z."):
            # Set the velocity sigma to be 1 km/s
            dimensional_std.append(10./299792458e-3)
            
        elif dimension.startswith("convolve."): 
            dimensional_std.append(0.1 * point.get(dimension))

        elif dimension.startswith("normalise."):

            if dimension.endswith(".bandwidth"):
                dimensional_std.append(1000.)
                continue
            elif dimension.endswith(".s_scale"):
                dimensional_std.append(0.1)
                continue

            channel = dimension.split(".")[1]

            """
            if dimension.endswith(".s"):
                # Spline treatment of continuum
                s = np.random.normal(ball_point[di], np.sqrt(2*ball_point[di]))
                dimensional_std.append(np.max([s, 0]))
            """
            coefficient = dimension.split(".")[2]

            if coefficient.startswith("k"): # Spline knots
                coefficient = int(coefficient[1:])

                # There are better ways to do this
                dimensional_std.append(1)

            else: 
                # Polynomial treatment of continuum
                
                coefficient = int(coefficient[1:])
                order = model.configuration["normalise"][channel]["order"]
                observed_channel = observed_spectra[model.channels.index(channel)]
                
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
                dispersion = observed_channel.disp.mean()
                flux_scale = 3. * np.sqrt(observed_channel.variance[np.isfinite(observed_channel.variance)].mean())
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

    # Write over Pb priors
    all_fluxes = np.array(list(chain(*[each.flux for each in observed_spectra])))
    all_fluxes = all_fluxes[np.isfinite(all_fluxes)]
    for i, dimension in enumerate(model.dimensions):
        if dimension == "Pb":
            p0[:, i] = np.abs(np.random.normal(0, 0.3, size=walkers))

        elif dimension == "Vb":
            p0[:, i] = np.random.normal(0, np.std(all_fluxes), size=walkers)**2

    # Write over normalisation priors if necessary
    for i, pi in enumerate(p0):

        # Model the flux, but don't normalise it.
        pi_parameters = dict(zip(model.dimensions, pi))
        for channel in model.channels:
            n = 0
            while "normalise.{channel}.c{n}".format(channel=channel, n=n) in pi_parameters.keys():
                pi_parameters["normalise.{channel}.c{n}".format(channel=channel, n=n)] = 0
                n += 1

            if n > 0:
                # Set the final coefficient as 1, so we end up having no normalisation
                pi_parameters["normalise.{channel}.c{n}".format(channel=channel, n=n-1)] = 1.

        if n > 0:
            try:
                model_channels = model(observations=observed_spectra, **pi_parameters)
            except: continue
            
            for channel, observed_channel, model_channel_flux in zip(model.channels, observed_spectra, model_channels):

                continuum = (observed_channel.flux + np.random.normal(0, 1e-12 + np.abs(np.sqrt(observed_channel.variance))))/model_channel_flux
                finite = np.isfinite(continuum)
                if np.sum(finite) == 0: break

                # Get some normalisation coefficients
                order = model.configuration["normalise"][channel]["order"]
                coefficients = np.polyfit(observed_channel.disp[finite], continuum[finite], order)
                
                # Write over the prior values
                for j, coefficient in enumerate(coefficients):
                    index = model.dimensions.index("normalise.{channel}.c{n}".format(channel=channel, n=j))
                    p0[i, index] = coefficient


    return p0


def __safe_log_probability(theta, model, observed_spectra):
    try:
        probability = log_probability(theta, model, observed_spectra)
    except:
        return (theta, -np.inf)
    else:
        return (theta, probability)

def eval_prior(model):

    env = { "locals": None, "globals": None, "__name__": None, "__file__": None,
        "__builtins__": None, "normal": random.normal, "uniform": random.uniform,
        "abs": abs, }

    dimensions = model.grid_points.dtype.names
    return dict(zip(dimensions, [eval(model.priors[dimension], env) for dimension in dimensions]))


def random_scattering(observed_spectra, model, initial_thetas=None):
    """
    Calculate likelihoods for randomly drawn theta points across the parameter space.

    Args:
        observed_spectra (list of Spectrum1D objects): The observed data.
        model (sick.models.Model object): The model class.
        initial_thetas (list-type or None): The theta points to sample. If none are
            provided then the number of randomly drawn points is determined by the
            model configuration `model.solver.initial_samples`
    """

    logger.info("Performing random scattering...")

    # Random scattering
    ta = time()
    samples = model.configuration["solver"]["initial_samples"]

    if initial_thetas is None:
   
        # Evaluate psi in serial, then map to parallel
        astrophysical_samples = (eval_prior(model) for _ in xrange(samples))

        pool = multiprocessing.Pool(model.configuration["solver"].get("threads", 1))
        try:
            scatter_func = utils.wrapper(initial_point, [model, observed_spectra])
            points = pool.map(scatter_func, astrophysical_samples)

            lnprob_func = utils.wrapper(log_probability, [model, observed_spectra])
            probabilities = pool.map(lnprob_func, points)

        except:
            logging.exception("Exception raised while doing random random_scattering")
            raise

        else:
            index = np.argmax(probabilities)
            p0 = points[index]

        finally:
            pool.close()
            pool.join()

    else:

        raise NotImplementedError
        if not isinstance(initial_thetas[0], (list, tuple, np.ndarray)):
            initial_thetas = [initial_thetas]

        [pool.apply_async(__safe_log_probability, args=(theta, model, observed_spectra), callback=callback) \
            for theta in initial_thetas]

        index = np.argmax([result[1] for result in results])
        p0 = results[index][0]

    logger.info("Calculating log probabilities of {0:.0f} prior points took {1:.2f} seconds".format(
        samples, time() - ta))
    
    return p0


def optimise(p0, observed_spectra, model, **kwargs):
    """
    Numerically optimise the likelihood from a provided point p0.

    Args:
        p0 (list-type): The theta point to optimise from.
        observed_spectra (list of Spectrum1D objects): The observed data.
        model (sick.models.Model object): The model class.

    Returns:
        opt_theta (list-type) the numerically optimised point.
    """

    logger.info("Optimising from point:")
    for dimension, value in zip(model.dimensions, p0):
        logger.info("  {0}: {1:.2e}".format(dimension, value))

    ta = time()

    full_output = "full_output" in kwargs and kwargs["full_output"]

    # Set some keyword defaults
    default_kwargs = {
        "maxfun": 1000,
        "xtol": 100,
        "ftol": 0.1
    }
    [kwargs.setdefault(k, v) for k, v in default_kwargs.iteritems()]
    
    # And we need to overwrite this one because we want all the information, even if the user doesn't.
    kwargs.update({"full_output": True})

    # Optimisation
    opt_theta, fopt, niter, funcalls, warnflag = optimize.fmin(
        lambda theta, model, obs: -log_probability(theta, model, obs), p0,
        args=(model, observed_spectra), **kwargs)

    if warnflag > 0:
        messages = [
            "Maximum number of function evaluations made. Optimised solution may be inaccurate.",
            "Maximum number of iterations reached. Optimised solution may be inaccurate."
        ]
        logger.warn(messages[warnflag - 1])
    logger.info("Optimisation took {0:.2f} seconds".format(time() - ta))

    if full_output:
        return (opt_theta, fopt, niter, funcalls, warnflag)

    return opt_theta


def sample(observed_spectra, model, p0=None, lnprob0=None, rstate0=None, burn=None, sample=None):
    """
    Set up an EnsembleSampler and sample the parameters given the model and data.
    """

    t_init = time()

    if not isinstance(model, models.Model):
        model = models.Model(model)
    model.map_channels(observed_spectra)

    # Set up MCMC settings and arrays
    walkers = model.configuration["solver"]["walkers"]
    if burn is None:
        burn = model.configuration["solver"]["burn"]
    if sample is None:
        sample = model.configuration["solver"]["sample"]

    mean_acceptance_fractions = np.zeros(burn + sample)
    autocorrelation_time = np.zeros((burn, len(model.dimensions)))

    # Initialise the sampler
    sampler = emcee.EnsembleSampler(walkers, len(model.dimensions), log_probability,
        args=(model, observed_spectra), threads=model.configuration["solver"].get("threads", 1))

    # Start sampling
    try:
        for i, (pos, lnprob, rstate) in enumerate(sampler.sample(p0, \
            lnprob0=lnprob0, rstate0=rstate0, iterations=burn)):
            
            mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)
            
            # Announce progress
            logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f}, maximum log probability"\
                " in last step was {2:.3e}".format(i + 1, mean_acceptance_fractions[i],
                    np.max(sampler.lnprobability[:, i])))

            if mean_acceptance_fractions[i] in (0, 1):
                raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(mean_acceptance_fractions[i]))

    except KeyboardInterrupt as e:
        # Convergence achieved.
        mean_acceptance_fractions = mean_acceptance_fractions[i + 1 + sample]

    # Save the chain and calculated log probabilities for later
    chain, lnprobability = sampler.chain, sampler.lnprobability

    logger.info("Resetting chain...")
    sampler.reset()

    logger.info("Sampling posterior...")
    for j, state in enumerate(sampler.sample(pos, iterations=sample)):
        mean_acceptance_fractions[i + j + 1] = np.mean(sampler.acceptance_fraction)

    # Concatenate the existing chain and lnprobability with the posterior samples
    chain = np.concatenate([chain, sampler.chain], axis=1)
    lnprobability = np.concatenate([lnprobability, sampler.lnprobability], axis=1)

    # Get the maximum likelihood theta
    ml_index = np.argmax(lnprobability.reshape(-1))
    ml_values = chain.reshape(-1, len(model.dimensions))[ml_index]

    # Get the quantiles
    posteriors = {}
    for parameter_name, ml_value, (quantile_16, quantile_84) in zip(model.dimensions, ml_values, 
        map(lambda v: (v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(sampler.chain.reshape(-1, len(model.dimensions)), [16, 50, 84], axis=0)))):
        posteriors[parameter_name] = (ml_value, quantile_16, quantile_84)

        # Transform redshift posteriors to velocity posteriors
        if parameter_name.startswith("z."):
            posteriors["v_rad." + parameter_name[2:]] = list(np.array(posteriors[parameter_name]) * 299792458e-3)

    # Send back additional information
    additional_info = {
        "priors": p0,
        "chain": chain,
        "lnprobability": lnprobability,
        "mean_acceptance_fractions": mean_acceptance_fractions,
        "time_elapsed": time() - t_init
    }
    return posteriors, sampler, additional_info


def solve(observed_spectra, model, initial_thetas=None, **kwargs):
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

        most_probable_scattered_point = random_scattering(observed_spectra, model, initial_thetas)

        kwargs_copy = kwargs.copy()
        kwargs_copy.update({"full_output": True})
        opt_theta, fopt, niter, funcalls, warnflag = optimise(most_probable_scattered_point, observed_spectra,
            model, **kwargs_copy)

        # Sample around opt_theta using some sensible things
        p0 = sample_ball(dict(zip(model.dimensions, opt_theta)), observed_spectra, model)
        #p0 = np.array([opt_theta + 1e-4 * random.randn(len(model.dimensions)) for each in range(walkers)])

    else:
        warnflag, p0 = 0, priors.explicit(model, observed_spectra)

    logger.info("Starting point summary:")
    for i, dimension in enumerate(model.dimensions):
        if len(p0.shape) > 1 and p0.shape[1] > 1:
            logger.info("  Parameter {0} - mean: {1:.4e}, std: {2:.4e}, min: {3:.4e}, max: {4:.4e}".format(
                dimension, np.mean(p0[:, i]), np.std(p0[:, i]), np.min(p0[:, i]), np.max(p0[:, i])))
        else:
            logger.info(" Parameter {0} - initial point: {1:.2e}".format(dimension, p0[i]))

    # Perform MCMC sampling
    posteriors, sampler, additional_info = sample(observed_spectra, model, p0)

    # Update the additional_info dictionary with information from other steps    
    additional_info.update({ 
        "warnflag": warnflag,
        "time_elapsed": time() - t_init
    })
    logger.info("Completed in {0:.2f} seconds".format(additional_info["time_elapsed"]))

    return (posteriors, sampler, additional_info)


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

    :param evaluated_priors:
        A dictionary containing the priors (values) to employ for the scattering
        process for each parameter (key)

    :type evaluated_priors:
        dict

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :param observations:
        The observed spectra.

    :type observations:
        iterable of :class:`sick.specutils.Spectrum1D` objects

    :raises ValueError:
        If the scattering distribution for a parameter is uninterpretable.

    :returns:
        An initial starting point for :math:`\\Theta`.

    :rtype:
        :class:`numpy.array`
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

    initial_distributions.update({"Vb": "normal({0}, 0.5 * {0})".format(np.std(all_fluxes)**2)})

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
    
    for i, parameter in enumerate(model.parameters):

        # Have we already evaluated this parameter?
        if parameter in evaluated_priors.keys():
            current_point.append(evaluated_priors[parameter])
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
                
                if "convolve.{0}".format(channel) in model.parameters:

                    # We have to evaluate this prior now
                    sigma = eval(initial_distributions["convolve.{0}".format(channel)], env)
                    kernel = (sigma/(2 * (2*np.log(2))**0.5))/np.mean(np.diff(model.dispersion[channel]))
                    
                    evaluated_priors["convolve.{0}".format(channel)] = sigma

                    # Smooth the model intensities
                    model_intensities[channel] = ndimage.gaussian_filter1d(model_intensities[channel], kernel)

                # Get continuum knots/coefficients for each aperture
                observed_channel = observations[model.channels.index(channel)]

                continuum = observed_channel.flux/np.interp(observed_channel.disp,
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

        # Is there an explicitly specified distribution for this parameter?
        specified_prior = initial_distributions.get(parameter, False)
        if specified_prior:
            # Evaluate the prior given the environment information
            current_point.append(eval(specified_prior, env))
            continue

        # These are all implicit priors from here onwards.
        if parameter.startswith("normalise."):

            # Get the coefficient
            channel = parameter.split(".")[1]
            coefficient_index = int(parameter.split(".")[2][1:])
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
            raise ValueError("Cannot interpret initial scattering distribution for {0}".format(parameter))

    # Check that we have the full number of walker values
    if len(current_point) == len(model.parameters):
        return current_point

    else:
        return current_point + [0.] * (len(model.parameters) - len(current_point))


def log_prior(theta, model):
    """
    Return the prior for a set of theta given the model.

    :param theta:
        The model parameter values.

    :type theta: list

    :param model:
        The model class.

    :type model: :class:`sick.models.Model`

    :returns:
        The logarithmic prior for the parameters theta.

    :rtype: float
    """

    log_prior = 0
    for parameter, value in zip(model.parameters, theta):

        if parameter in model.priors:
            f = eval(model.priors[parameter], _prior_eval_env_)
            log_prior += f(value)

        # Check smoothing values
        if parameter.startswith("convolve.") and 0 > value:
            return -np.inf

        # Check for outlier parameters
        if parameter == "Pb" and not (1. > value > 0.) \
        or parameter == "Vb" and 0 > value:
            return -np.inf

    logging.debug("Returning log-prior of {0:.2e} for parameters: {1}".format(
        log_prior, ", ".join(["{0} = {1:.2e}".format(name, value) \
            for name, value in zip(model.parameters, theta)])))
    return log_prior


def log_likelihood(theta, model, observations):
    """
    Return the logarithmic likelihood for a set of theta given the model.

    :param theta:
        The model parameter values.

    :type theta: list

    :param model:
        The model class.

    :type model: :class:`sick.models.Model`

    :param observations:
        The observed spectra.

    :type observations:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :returns:
        The logarithmic likelihood for the parameters theta.

    :rtype: float
    """

    theta_dict = dict(zip(model.parameters, theta))

    try:
        model_fluxes, model_continua = model(observations=observations,
            full_output=True, **theta_dict)
    except:
        return -np.inf

    likelihood = 0
    for (channel, model_flux, model_continuum, observed_spectrum) \
    in zip(model.channels, model_fluxes, model_continua, observations):

        signal_inverse_variance = 1.0/(observed_spectrum.variance \
            + model_flux**2 * np.exp(2. * theta_dict["f.{0}".format(channel)]))

        signal_likelihood = -0.5 * ((observed_spectrum.flux - model_flux)**2 \
            * signal_inverse_variance - np.log(signal_inverse_variance))

        # Are we modelling the outliers as well?
        if "Pb" in theta_dict.keys():
            outlier_inverse_variance = 1.0/(theta_dict["Vb"] + observed_spectrum.variance \
                + model_flux**2 * np.exp(2. * theta_dict["f.{0}".format(channel)]))

            outlier_likelihood = -0.5 * ((observed_spectrum.flux - model_continuum)**2 \
                * outlier_inverse_variance - np.log(outlier_inverse_variance))

            Pb = theta_dict["Pb"]
            finite = np.isfinite(outlier_likelihood * signal_likelihood)
            likelihood += np.sum(np.logaddexp(
                np.log(1. - Pb) + signal_likelihood[finite],
                np.log(Pb) + outlier_likelihood[finite]))

        else:
            finite = np.isfinite(signal_likelihood)
            likelihood += np.sum(signal_likelihood[finite])
    if likelihood == 0:
        return -np.inf

    logger.debug("Returning log-likelihood of {0:.2e} for parameters: {1}".format(
        likelihood, ", ".join(["{0} = {1:.2e}".format(name, value) \
            for name, value in theta_dict.iteritems()])))  
    return likelihood


def log_probability(theta, model, observations):
    """
    Return the logarithmic probability (prior + likelihood) for theta given the data.

    :param theta:
        The model parameter values.

    :type theta: list

    :param model:
        The model class.

    :type model: :class:`sick.models.Model`

    :param observations:
        The observed spectra.

    :type observations:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :returns:
        The logarithmic probability for the parameters theta.

    :rtype: float
    """

    prior = log_prior(theta, model)
    if np.isinf(prior):
        logger.debug("Returning -inf log-probability because log-prior was -inf")
        return prior

    return prior + log_likelihood(theta, model, observations)


def sample_ball(point, observed_spectra, model):
    """
    Return a multi-parameteral Gaussian around a ball point.

    :param point:
        The point to create the ball around.

    :type point:
        dict

    :param observed_spectra:
        The observed spectra.

    :type observed_spectra:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :returns:
        An array of starting values for the walkers.
    """ 
    
    # Create a sample ball around the result point
    ball_point = [point.get(parameter, 0) for parameter in model.parameters]
    
    parameteral_std = []
    jitter_indices = []
    for di, parameter in enumerate(model.parameters):

        if parameter in model.grid_points.dtype.names:
            # Set sigma to be 30% of the parameter dynamic range
            parameteral_std.append(0.05 * np.ptp(model.grid_boundaries[parameter]))
           
        elif parameter.startswith("z."):
            # Set the velocity sigma to be 1 km/s
            parameteral_std.append(10./299792458e-3)
            
        elif parameter.startswith("convolve."): 
            parameteral_std.append(0.1 * point.get(parameter))

        elif parameter.startswith("normalise."):

            channel = parameter.split(".")[1]

            """
            if parameter.endswith(".s"):
                # Spline treatment of continuum
                s = np.random.normal(ball_point[di], np.sqrt(2*ball_point[di]))
                parameteral_std.append(np.max([s, 0]))
            """
            coefficient = parameter.split(".")[2]

            if coefficient.startswith("k"): # Spline knots
                coefficient = int(coefficient[1:])

                # There are better ways to do this
                parameteral_std.append(1)

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
                parameteral_std.append(flux_scale/(dispersion**(order - coefficient)))

        else:
            # Jitter
            parameteral_std.append(0.1)
            jitter_indices.append(di)
            
    walkers = model.configuration["settings"]["walkers"]
    p0 = emcee.utils.sample_ball(ball_point, parameteral_std, size=walkers)

    # Write over jitter priors
    for ji in jitter_indices:
        p0[:, ji] = np.random.uniform(-10, 1, size=walkers)

    # Write over Pb priors
    all_fluxes = np.array(list(chain(*[each.flux for each in observed_spectra])))
    all_fluxes = all_fluxes[np.isfinite(all_fluxes)]
    for i, parameter in enumerate(model.parameters):
        if parameter == "Pb":
            p0[:, i] = np.abs(np.random.normal(0, 0.3, size=walkers))

        elif parameter == "Vb":
            p0[:, i] = np.random.normal(0, np.std(all_fluxes), size=walkers)**2

    # Write over normalisation priors if necessary
    for i, pi in enumerate(p0):

        # Model the flux, but don't normalise it.
        pi_parameters = dict(zip(model.parameters, pi))
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
                    index = model.parameters.index("normalise.{channel}.c{n}".format(channel=channel, n=j))
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

    parameters = model.grid_points.dtype.names
    return dict(zip(parameters, [eval(model.priors[parameter], env) for parameter in parameters]))


def random_scattering(observed_spectra, model):
    """
    Calculate likelihoods for randomly drawn theta points across the parameter space.

    :param observed_spectra:
        The observed spectra.

    :type observed_spectra:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :returns:
        The most probable starting point :math:`\Theta` for optimisation.
    """

    logger.info("Performing random scattering...")

    # Random scattering
    ta = time()
    samples = model.configuration["settings"]["initial_samples"]

    # Evaluate some initial thetas psi in serial, then map to parallel
    astrophysical_samples = (eval_prior(model) for _ in xrange(samples))

    pool = multiprocessing.Pool(model.configuration["settings"].get("threads", 1))
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

    logger.info("Calculating log probabilities of {0:.0f} prior points took {1:.2f} seconds".format(
        samples, time() - ta))
    
    return p0


def optimise(p0, observed_spectra, model, **kwargs):
    """
    Numerically optimise the likelihood from a provided point p0.

    :param p0:
        The initial starting point :math:`\Theta` to optimise from.

    :type p0:
        :class:`numpy.array`

    :param observed_spectra:
        The observed spectra.

    :type observed_spectra:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :param kwargs:
        Keyword arguments to pass directly to the optimisation call.

    :type kwargs:
        dict

    :returns:
        The numerically optimised point :math:`\Theta_{opt}`.
    """

    logger.info("Optimising from point:")
    for parameter, value in zip(model.parameters, p0):
        logger.info("  {0}: {1:.2e}".format(parameter, value))

    ta = time()

    full_output = "full_output" in kwargs and kwargs["full_output"]

    # Set some keyword defaults
    default_kwargs = {
        "maxfun": 10000,
        "maxiter": 10000,
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


def sample(observed_spectra, model, p0=None, lnprob0=None, rstate0=None, burn=None,
    sample=None):
    """
    Set up an EnsembleSampler and sample the parameters given the model and data.

    :param observed_spectra:
        The observed spectra.

    :type observed_spectra:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :param p0:
        The starting point for all the walkers.

    :type p0:
        :class:`numpy.ndarray`

    :param lnprob0: [optional]
        The log posterior probabilities for the walkers at positions given by
        ``p0``. If ``lnprob0`` is None, the initial values are calculated.

    :type lnprob0:
        :class:`numpy.ndarray`

    :param rstate0: [optional]
        The state of the random number generator.

    :param burn: [optional]
        The number of samples to burn. Defaults to the ``burn`` value in
        ``settings`` of ``model.configuration``.

    :type burn:
        int

    :param sample: [optional]
        The number of samples to make from the posterior. Defaults to the 
        ``sample`` value in ``settings`` of ``model.configuration``.

    :type sample:
        int

    :returns:
        A tuple containing the posteriors, sampler, and general information.
    """

    t_init = time()

    if not isinstance(model, models.Model):
        model = models.Model(model)
    model.map_channels(observed_spectra)

    # Set up MCMC settings and arrays
    walkers = model.configuration["settings"]["walkers"]
    if burn is None:
        burn = model.configuration["settings"]["burn"]
    if sample is None:
        sample = model.configuration["settings"]["sample"]

    mean_acceptance_fractions = np.zeros(burn + sample)
    autocorrelation_time = np.zeros((burn, len(model.parameters)))

    # Initialise the sampler
    sampler = emcee.EnsembleSampler(walkers, len(model.parameters),
        log_probability, args=(model, observed_spectra),
        threads=model.configuration["settings"].get("threads", 1))

    # Start sampling
    try:
        for i, (pos, lnprob, rstate) in enumerate(sampler.sample(p0, \
            lnprob0=lnprob0, rstate0=rstate0, iterations=burn)):
            
            mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)
            
            # Announce progress
            logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f},"\
                " maximum log probability in last step was {2:.3e}".format(i + 1,
                mean_acceptance_fractions[i], np.max(sampler.lnprobability[:, i])))

            if mean_acceptance_fractions[i] in (0, 1):
                raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(
                    mean_acceptance_fractions[i]))

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
    ml_values = chain.reshape(-1, len(model.parameters))[ml_index]

    # AGAINST MY BETTER JUDGEMENT:
    # Calculate a reduced chi-sq value for the most likely theta.
    ml_model_fluxes = model(observations=observed_spectra, **dict(zip(model.parameters, ml_values)))
    r_chi_sq, num_pixels = 0, 0
    for observed_spectrum, model_flux in zip(observed_spectra, ml_model_fluxes):
        chi_sq = (observed_spectrum.flux - model_flux)**2/observed_spectrum.variance
        r_chi_sq += np.nansum(chi_sq)
        num_pixels += np.sum(np.isfinite(chi_sq))
    r_chi_sq /= (num_pixels - len(model.parameters) - 1)

    # Get the quantiles
    posteriors = {}
    for parameter_name, ml_value, (quantile_16, quantile_84) in zip(model.parameters, ml_values, 
        map(lambda v: (v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(sampler.chain.reshape(-1, len(model.parameters)), [16, 50, 84], axis=0)))):
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
        "time_elapsed": time() - t_init,
        "reduced_chi_sq": r_chi_sq
    }
    return posteriors, sampler, additional_info


def solve(observed_spectra, model, **kwargs):
    """
    Solve for the model parameters theta given the observed spectra.

    :param observed_spectra:
        The observed spectra.

    :type observed_spectra:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :returns:
        A 3-length tuple containing the posteriors, sampler, and general information.
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
    if model.configuration["settings"].get("optimise", True):

        most_probable_scattered_point = random_scattering(observed_spectra, model)

        kwargs_copy = kwargs.copy()
        kwargs_copy.update({"full_output": True})
        opt_theta, fopt, niter, funcalls, warnflag = optimise(
            most_probable_scattered_point, observed_spectra, model, **kwargs_copy)

        # Sample around opt_theta using some sensible things
        p0 = sample_ball(dict(zip(model.parameters, opt_theta)), observed_spectra, model)
        
    else:
        warnflag, p0 = 0, priors.explicit(model, observed_spectra)

    logger.info("Starting point summary:")
    for i, parameter in enumerate(model.parameters):
        if len(p0.shape) > 1 and p0.shape[1] > 1:
            logger.info("  Parameter {0} - mean: {1:.4e}, std: {2:.4e}, min: {3:.4e}, max: {4:.4e}".format(
                parameter, np.mean(p0[:, i]), np.std(p0[:, i]), np.min(p0[:, i]), np.max(p0[:, i])))
        else:
            logger.info(" Parameter {0} - initial point: {1:.2e}".format(parameter, p0[i]))

    # Perform MCMC sampling
    posteriors, sampler, additional_info = sample(observed_spectra, model, p0)

    # Update the additional_info dictionary with information from other steps    
    additional_info.update({ 
        "warnflag": warnflag,
        "time_elapsed": time() - t_init
    })
    logger.info("Completed in {0:.2f} seconds".format(additional_info["time_elapsed"]))

    return (posteriors, sampler, additional_info)


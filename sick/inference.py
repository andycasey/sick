# coding: utf-8

""" Prior, likelihood, and probability functions for inference """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["log_prior", "log_likelihood", "log_probability"]

import logging
from functools import partial

import numpy as np
from scipy import stats

logger = logging.getLogger("sick")

_prior_eval_env_ = { 
    "locals": None,
    "globals": None,
    "__name__": None,
    "__file__": None,
    "__builtins__": None,
    "uniform": lambda a,b: partial(stats.uniform.logpdf, **{"loc":a,"scale":b-a}),
    "normal": lambda a,b: partial(stats.norm.logpdf, **{"loc":a,"scale":b})
}

def log_prior(theta, model):
    """
    Return the prior for a set of theta given the model.

    :param theta:
        The model parameter values.

    :type theta:
        list

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :returns:
        The logarithmic prior for the parameters theta.

    :rtype:
        float
    """


    log_prior = 0
    for parameter, value in zip(model.parameters, theta):

        # These non-physical priors are hard-coded
        if (parameter[:9] == "convolve." and 0 > value) \
        or (parameter == "Pb" and not (1. > value > 0.)) \
        or (parameter == "Vb" and 0 > value):
            return -np.inf

        try:
            prior = model.priors[parameter]

        except KeyError:
            continue

        else:
            f = eval(prior, _prior_eval_env_)
            log_prior += f(value)

    logging.debug("Returning log prior of {0:.2e} for parameters: {1}".format(
        log_prior, ", ".join(["{0} = {1:.2e}".format(name, value) \
            for name, value in zip(model.parameters, theta)])))
    return log_prior



def log_likelihood(theta, model, data):
    """
    Return the logarithmic likelihood for a set of theta given the model.

    :param theta:
        The model parameter values.

    :type theta:
        list

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :param data:
        The observed spectra.

    :type data:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :returns:
        The logarithmic likelihood for the parameters theta.

    :rtype:
        float
    """

    theta_dict = dict(zip(model.parameters, theta))
    try:
        model_fluxes, model_continua = model(data=data,
            full_output=True, **theta_dict)
    except ValueError:
        return -np.inf

    likelihood, num_finite_pixels = 0, 0
    for (channel, model_flux, model_continuum, observed_spectrum) \
    in zip(model.channels, model_fluxes, model_continua, data):

        # Underestimated variance?
        if "f.{}".format(channel) in theta_dict:
            additional_noise = model_flux * np.exp(2. * theta_dict["f.{0}".format(channel)])
            signal_inverse_variance = 1.0/(observed_spectrum.variance + additional_noise)
        else:
            additional_noise = 0.
            signal_inverse_variance = observed_spectrum.ivariance

        signal_likelihood = -0.5 * ((observed_spectrum.flux - model_flux)**2 \
            * signal_inverse_variance - np.log(signal_inverse_variance))

        # Are we modelling the outliers as well?
        if "Pb" in theta_dict:
            outlier_inverse_variance = 1.0/(theta_dict["Vb"] + observed_spectrum.variance \
                + additional_noise)
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
        num_finite_pixels += finite.sum()

    if likelihood == 0:
        return -np.inf

    logger.debug("Returning log-likelihood of {0:.2e} with {1:.0f} pixels for "\
        "parameters: {2}".format(likelihood, num_finite_pixels, 
            ", ".join(["{0} = {1:.2e}".format(name, value) \
                for name, value in theta_dict.iteritems()])))  
    return likelihood


def log_probability(theta, model, data):
    """
    Return the logarithmic probability for theta given the data.

    :param theta:
        The model parameter values.

    :type theta:
        list

    :param model:
        The model class.

    :type model:
        :class:`sick.models.Model`

    :param data:
        The observed spectra.

    :type data:
        A list of :class:`sick.specutils.Spectrum1D` objects.

    :returns:
        The logarithmic probability for the parameters theta.

    :rtype:
        float
    """

    prior = log_prior(theta, model)
    if np.isinf(prior):
        return prior
    return prior + log_likelihood(theta, model, data)

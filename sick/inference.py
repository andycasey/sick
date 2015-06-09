#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Inference and objective functions. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import numpy as np
from functools import partial

from scipy import stats

logger = logging.getLogger("sick")

_ = "locals globals __name__ __file__ __builtins__".split()
_prior_eval_env_ = dict(zip(_, [None] * len(_)))
_prior_eval_env_.update({
    "uniform": lambda a, b: partial(stats.uniform.logpdf,
        **{ "loc": a, "scale": b - a }),
    "normal": lambda a, b: partial(stats.norm.logpdf, 
        **{ "loc": a, "scale": b })
})

def ln_likelihood(theta, model, data, debug=False, **kwargs):

    logger.debug("In likelihood func with {}".format(theta))

    sigma_clip = model._configuration.get("settings", {}).get("sigma_clip", -1)

    try:
        # Setting debug to True means it will re-raise any exceptions when
        # trying to approximate spectra from the grid.
        model_fluxes, model_variances, channels, continua = model(theta, data,
            debug=True, full_output=True, __return_continuum=True, **kwargs)
    
    except:
        logger.exception("Returning -inf for {} because the model data couldn't"
            "be generated due to the following exception:".format(theta))
        if debug: raise
        return -np.inf

    ln_likelihood, num_pixels = 0, 0
    for channel, spectrum, model_flux, model_variance, continuum \
    in zip(channels, data, model_fluxes, model_variances, continua):
        if channel is None: # no finite model fluxes
            continue 

        # Observed and model variance (where it exists)
        variance = spectrum.variance + model_variance * continuum**2

        # Any on-the-fly sigma-clipping?
        if sigma_clip > 0:
            chi_sq = (spectrum.flux - model_flux)**2 / variance
            mask = chi_sq > sigma_clip**2
            logger.debug("Num masking due to sigma clipping: {0} in {1}".format(
                mask.sum(), channel))
            if float(mask.sum()/variance.size) < 0.05:
                variance[mask] = np.nan

        # Any underestimated variance?
        ln_f = theta.get("ln_f", theta.get("ln_f_{}".format(channel), None))
        if ln_f is not None:
            variance += model_flux**2 * np.exp(2.0 * ln_f)

        # Calculate pixel likelihoods.
        ivar = 1.0/variance
        likelihood = -0.5 * ((spectrum.flux - model_flux)**2 * ivar \
            - np.log(ivar))

        # Only allow for positive flux to be produced!
        pixels = np.isfinite(likelihood) * (model_flux > 0)
    
        # Outliers?
        if "Po" in theta:
            # Calculate outlier likelihoods.
            outlier_ivar = 1.0/(variance + theta["Vo"])
            outlier_likelihood = -0.5 * ((spectrum.flux - continuum)**2 \
                * outlier_ivar - np.log(outlier_ivar))

            Po = theta["Po"]
            pixels *= np.isfinite(outlier_likelihood)
            ln_likelihood += np.sum(np.logaddexp(
                np.log(1. - Po) + likelihood[pixels],
                np.log(Po) + outlier_likelihood[pixels]))

        else:
            ln_likelihood += np.sum(likelihood[pixels])

        num_pixels += pixels.sum()

    if num_pixels == 0:
        logger.debug("No pixels used for likelihood calculation! Returning -inf")
        return -np.inf

    if not np.isfinite(ln_likelihood):
        raise WTFError("non-finite likelihood!")

    logger.debug("Returning ln(L) = {0:.1e} with {1} pixels from {2}".format(
        ln_likelihood, num_pixels, theta))
    return ln_likelihood


def ln_prior(theta, model, debug=False):

    # Need to calculate priors if there is:
    #   - the model parameter is resolution_* and negative
    for resolution_parameter in model._resolution_parameters:
        if 0 > theta.get(resolution_parameter, 1):
            logger.debug("Retuning log prior of -inf (bad {0} value) for theta"\
                " parameters: {1}".format(resolution_parameter, theta))
            return -np.inf
    
    # Outlier parameters:
    #   - The parameter is Po and it is not (1 > value > 0)
    #   - The parameter is Vo and not > 0
    if not 1 > theta.get("Po", 0.5) > 0 or 0 >= theta.get("Vo", 1):
        logger.debug("Returning log prior of -inf (bad Po/Vo value) for theta "\
            "parameters: {0}".format(theta))
        return -np.inf

    #   - prior specified for that model parameter
    ln_prior = 0
    for parameter, rule in model._configuration.get("priors", {}).items():
        if parameter not in theta or not rule: continue

        try:
            f = eval(rule, _prior_eval_env_)
            ln_prior += f(theta[parameter])

        except:
            logger.exception("Failed to evaluate prior for {0}: {1}".format(
                parameter, rule))
            if debug: raise

    logger.debug("Returning log prior of {0:.2e} for parameters: {1}".format(
        ln_prior, theta))

    return ln_prior


def _ln_probability(theta_dict, model, data, debug, **kwargs):
    prior = ln_prior(theta_dict, model, debug=debug)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(theta_dict, model, data, debug=debug, **kwargs)


def ln_probability(theta, parameters, model, data, debug=False, **kwargs):
    theta_dict = dict(zip(parameters, theta))
    return _ln_probability(theta_dict, model, data, debug, **kwargs)

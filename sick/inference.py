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

_prior_eval_env_ = { 
    "locals": None,
    "globals": None,
    "__name__": None,
    "__file__": None,
    "__builtins__": None,
    "uniform": lambda a, b: partial(stats.uniform.logpdf,
        **{ "loc": a, "scale": b - a} ),
    "normal": lambda a, b: partial(stats.norm.logpdf, 
        **{ "loc": a, "scale": b })
}

def ln_likelihood(theta, model, data, debug=False):

    logger.debug("In likelihood func with {}".format(theta))

    try:
        # Setting debug to True means it will re-raise any exceptions when
        # trying to approximate spectra from the grid.
        model_fluxes, model_variances, channels, continua = model(theta, data,
            debug=True, full_output=True, __return_continuum=True)
    
    except:
        logger.exception("Returning -inf for {} because the model data couldn't"
            "be generated due to the following exception:".format(theta))
        if debug: raise
        return -np.inf

    # any additional variance?
    # mixture model?

    ln_likelihood, sum_pixels = 0, 0
    for channel, spectrum, model_flux, model_variance, continuum \
    in zip(channels, data, model_fluxes, model_variances, continua):
        if channel is None: # no finite model fluxes
            continue 

        variance = spectrum.variance + continuum * model_variance
        # TODO Is there underestimated variance?
        ln_f = theta.get("f", theta.get("f_{}".format(channel), None))
        if ln_f is not None:
            variance += model_flux**2 * np.exp(2.0 * ln_f)
            ivar = 1.0/variance

        else:
            ivar = 1.0/variance

        # TODO: outlier modelling

        # Calculate likelihoods
        likelihoods = -0.5 * (model_flux - spectrum.flux)**2 * ivar \
            - 0.5 * np.log(variance)
        pixels = np.isfinite(likelihoods)

        ln_likelihood += np.sum(likelihoods[pixels])
        sum_pixels += pixels.sum()

    if sum_pixels == 0:
        logger.debug("No pixels used for likelihood calculation! Returning -inf")
        return -np.inf

    if not np.isfinite(ln_likelihood):
        raise WTFError("non-finite likelihood!")

    logger.debug("Returning ln(L) = {0:.1e} with {1} pixels from {2}".format(
        ln_likelihood, sum_pixels, theta))
    return ln_likelihood



def ln_prior(theta, model):

    ln_prior = 0
    priors = model._configuration.get("priors", {})
    for parameter, value in theta.items():

        # Sensible priors on resolution, Pb, Vb.
        if 0 > value and parameter.startswith("resolution_") \
        or (parameter == "Pb" and not (1 > value > 0)) \
        or (parameter == "Vb" and 0 > value):
            return -np.inf

        # Evaluate the prior.
        rule = priors.get(parameter, None)
        if rule:
            try:
                f = eval(rule, _prior_eval_env_)
                ln_prior += f(value)
            
            except:
                logger.exception("Failed to evaluate prior for {0}: {1}".format(
                    parameter, rule))
                raise

        logger.debug("Returning log prior of {0:.2e} for parameters: {1}"\
            .format(ln_prior, theta))

    return ln_prior


def _ln_probability(theta_dict, model, data, debug):
    prior = ln_prior(theta_dict, model)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(theta_dict, model, data, debug=debug)


def ln_probability(theta, parameters, model, data, debug=False):
    theta_dict = dict(zip(parameters, theta))
    return _ln_probability(theta_dict, model, data, debug)

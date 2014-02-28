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
from collections import OrderedDict

# Third-party
import emcee
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import config, models, utils, specutils

logger = logging.getLogger(__name__)

__all__ = ['analyze', 'analyze_star', 'chi_squared_fn']


def initialise_priors(configuration, model, observed_spectra, aperture_mapping, nwalkers=1):
    """ Initialise the priors (or initial conditions) for the analysis """

    walker_priors = []
    measured_doppler_shifts = {}

    ordered_parameter_names = configuration["priors"].keys()

    # Jitter is a special parameter, and cannot be used as a name for any model parameters
    if "jitter" in ordered_parameter_names:
        raise ValueError("jitter cannot be used as a model parameter name as it is reserved for MCMC ensemble samplers")

    for i in xrange(nwalkers):

        current_walker = []
        for j, parameter_name in enumerate(ordered_parameter_names):
            if parameter_name == "jitter": continue

            prior_value = configuration["priors"][parameter_name]

            try:
                prior_value = float(prior_value)
                if i == 0:
                    logging_level = logging.info if nwalkers == 1 else logging.warn
                    logging_level("Initialised {0} parameter as a single value: {1:.2e}".format(
                        parameter_name, prior_value))

            except:

                # We probably need to evaluate this.
                if prior_value.lower() == "uniform":
                    # Only works on stellar parameter values.

                    index = model.colnames.index(parameter_name)
                    possible_points = model.grid_points[:, index]

                    if i == 0:
                        logging.info("Initialised {0} parameter with uniform distribution between {1:.2e} and {2:.2e}".format(
                            parameter_name, np.min(possible_points), np.max(possible_points)))
                    current_walker.append(random.uniform(np.min(possible_points), np.max(possible_points)))

                elif prior_value.lower().startswith("cross_correlate") \
                    and parameter_name.lower().startswith("doppler_correct.") and parameter_name.lower().endswith(".perform"):

                    # Do we need to measure the velocity?
                    aperture = parameter_name.split(".")[1]
                    if aperture not in measured_doppler_shifts:

                        # Measure the velocity
                        template_filename, region = literal_eval(prior_value.lstrip("cross_correlate"))

                        velocity, velocity_err = observed_spectra[aperture_mapping.index(aperture)].cross_correlate(
                            specutils.Spectrum1D.load(template_filename),
                            region
                            )

                        # Safeguards against bad velocity measurements?
                        if np.abs(velocity) > 500:
                            logging.warn("Measured absolute velocity in {0} aperture is larger than 500 km/s, assuming uniformed prior (0 km/s +/- 100 km/s)".format(aperture))
                            measured_doppler_shifts[aperture] = (0, 100)

                        else:
                            measured_doppler_shifts[aperture] = (velocity, velocity_err)

                    # Get the mu and sigma
                    mu, sigma = measured_doppler_shifts[aperture]

                    # Set mu to be negative so that it will correct for this doppler shift, not apply an additional shift
                    mu = -mu

                    if i == 0:
                        logging.info("Initialised {0} parameter with a normal distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
                            parameter_name, mu, sigma))
                    current_walker.append(random.normal(mu, sigma))

                elif prior_value.lower().startswith("normal"):
                    mu, sigma = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                    if i == 0:
                        logging.info("Initialised {0} parameter with a normal distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
                            parameter_name, mu, sigma))
                    current_walker.append(random.normal(mu, sigma))

                elif prior_value.lower().startswith("uniform"):
                    minimum, maximum = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                    if i == 0:
                        logging.info("Initialised {0} parameter with a uniform distribution between {1:.2e} and {2:.2e}".format(
                            parameter_name, minimum, maximum))
                    current_walker.append(random.uniform(minimum, maximum))

                else:
                    raise TypeError("prior type not valid for {parameter_name}".format(parameter_name=parameter_name))

            else:
                current_walker.append(prior_value)

        # Add the walker
        if nwalkers == 1:
            walker_priors = current_walker
        
        else:
            
            # Add jitter
            if ordered_parameter_names[-1] != "jitter":
                ordered_parameter_names.append("jitter")

            if i == 0:
                logging.info("Initialised jitter parameter with a uniform distribution between 0 and 1")
            current_walker.append(random.rand())
            walker_priors.append(current_walker)

    walker_priors = np.array(walker_priors)

    logging.info("Priors summary:")
    for i, ordered_parameter_name in enumerate(ordered_parameter_names):
        if len(walker_priors.shape) > 1 and walker_priors.shape[1] > 1:
            logging.info("\tParameter {0} - mean: {1:.2e}, min: {2:.2e}, max: {3:.2e}".format(
                ordered_parameter_name, np.mean(walker_priors[:, i]), np.min(walker_priors[:, i]), np.max(walker_priors[:, i])))
        else:
            logging.info("\tParameter {0} - initial point: {1:.2e}".format(
                ordered_parameter_name, walker_priors[i]))

    return (ordered_parameter_names, walker_priors)


def solve(observed_spectra, configuration, initial_guess=None):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    configuration : dict
        The configuration settings for this analysis.

    """

    if isinstance(configuration, str) and os.path.exists(configuration):
        configuration = config.load(configuration)

    # Check observed arms do not overlap
    observed_dispersions = [spectrum.disp for spectrum in observed_spectra]
    overlap = utils.find_spectral_overlap(observed_dispersions)
    if overlap is not None:
        raise ValueError("observed apertures cannot overlap in wavelength, but they do near {wavelength} Angstroms"
            .format(wavelength=overlap))

    # Load our model
    model = models.Model(configuration)

    # Get the aperture mapping from observed spectra to model spectra
    # For example, which index in our list of spectra corresponds to
    # 'blue', or 'red' in our model 
    aperture_mapping = model.map_apertures(observed_dispersions)

    
    # Make fmin_powell the default
    if "solution_method" not in configuration or configuration["solution_method"] == "fmin_powell":

        parameter_names, p0 = initialise_priors(configuration, model, observed_spectra, aperture_mapping)

        parameters_final = scipy.optimize.fmin_powell(chi_sq, p0,
            args=(parameter_names, observed_spectra, model), xtol=0.001, ftol=0.001)


    elif configuration["solution_method"] == "emcee":

        # Ensure we have the number of walkers and steps specified in the configuration
        nwalkers, nsteps = configuration["emcee"]["nwalkers"], configuration["emcee"]["nsteps"]
        lnprob0, rstate0 = None, None
        mean_acceptance_fractions = np.zeros(nsteps)
        
        # Use a callback if we are running in serial
        if configuration.get("threads", 1) == 1:
            sample_data = []
            def lnprob_fn_callback(parameters, chi_squared, log_likelihood):
                sample_point = list(parameters) + [chi_squared, log_likelihood]
                sample_data.append(sample_point)

        # Initialise priors and set up arguments for optimization
        parameter_names, p0 = initialise_priors(configuration, model, observed_spectra, aperture_mapping, nwalkers)
        optimisation_args = (parameter_names, observed_spectra, aperture_mapping, model, configuration, 
            lnprob_fn_callback, chi_squared_fn_callback)   

        logging.info("All priors initialsed for {0} walkers. Parameter names are: {1}".format(nwalkers, ", ".join(parameter_names)))
        sampler = emcee.EnsembleSampler(nwalkers, len(parameter_names), lnprob_fn, args=optimisation_args,
            threads=configuration.get("threads", 1))

        # Sample_data contains all the inputs, and the \chi^2 and L 
        # sampler_state = (pos, lnprob, state[, blobs])
        for i, sampler_state in enumerate(sampler.sample(
            p0, lnprob0=lnprob0, rstate0=rstate0, iterations=nsteps)):

            fraction_complete = (i + 1)/nsteps
            mean_acceptance_fraction = np.mean(sampler.acceptance_fraction)

            # Announce progress
            logging.info("Sampler is {0:.2f}% complete (step {1:.0f}) with a mean acceptance fraction of {2:.3f}".format(
                fraction_complete * 100, i + 1, mean_acceptance_fraction))

            mean_acceptance_fractions[i] = mean_acceptance_fraction
            if mean_acceptance_fraction == 0:
                logging.warn("Mean acceptance fraction is zero. Breaking out of MCMC!")
                break

        # Convert state to posteriors
        logging.info("The final mean acceptance fraction is {0:.3f}".format(mean_acceptance_fraction))

        # Blobs contain all the parameters sampled, chi_sq value and log-likelihood value

        if configuration.get("threads", 1) == 1:
            sampled_parameters = np.array(sample_data)

        else:
            sampled_parameters = np.array(sampler.blobs)


            sampled_parameters = sampled_parameters.reshape(
                sampled_parameters.shape[0] * sampled_parameters.shape[1], sampled_parameters.shape[2])

        # Get the most probable sampled point
        most_probable_index = np.argmax(sampled_parameters[:, -1])
        chi_sq, log_likelihood = sampled_parameters[most_probable_index, -2:]
        
        if not np.isfinite(log_likelihood):
            # TODO should we raise ModelError? or something?
            # You should probably check your configuration file for something peculiar
            raise a
        
        output = CallbackClass()
        final_callback = lambda *x: setattr(output, "data", x)
        chi_squared_fn(sampled_parameters[most_probable_index, :-2], parameter_names, observed_spectra,
            aperture_mapping, model, configuration, -np.inf, final_callback)

        try:
            chi_sq_returned, num_dof, posteriors, observed_spectra, model_spectra, masks = output.data

        except AttributeError:
            raise

        else:

            assert chi_sq == chi_sq_returned

            logging.info("Most probable values with a $\chi^2$ = {0:.2f} (N_dof = {1}, $\chi_r^2$ = {2:.2f}) and $L$ = {3:.4e}: {4}".format(
            chi_sq, num_dof, chi_sq/num_dof, log_likelihood,
            ", ".join(["{0} = {1:.2e}".format(p, v) \
                for p, v in zip(parameter_names, sampled_parameters[most_probable_index, :-2])])))

            # Save the state information
            state = {
                "chi_sq": chi_sq,
                "num_dof": num_dof,
                "pos": pos,
                "lnprob": lnprob,
                "state": state,
                "sampled_parameters": sampled_parameters,
                "log_likelihood": log_likelihood,
                "mean_acceptance_fractions": mean_acceptance_fractions,
            }
            return (posteriors, state, observed_spectra, model_spectra, masks)
            
    else:
        raise NotImplementedError

    # The following only occurs if we did not sample The Right Way(tm)
    # We will need to sample the \chi^2 function again with a callback to save
    # the results
    output = CallbackClass()
    final_callback = lambda *x: setattr(output, 'data', x)
    chi_squared_fn(parameters_final, parameter_names, observed_spectra, aperture_mapping, model, configuration, fail_value, final_callback)
    
    chi_sq, num_dof, posteriors, observed_spectra, model_spectra, masks = output.data

    # Create the state information
    state = {
        "chi_sq": chi_sq,
        "num_dof": num_dof,
    }
    return (posteriors, state, observed_spectra, model_spectra, masks)



def likelihood(parameters, parameter_names, observations, model, callback=None):
    """Calculates the likelihood that a given set of observations
    and parameters match the input models.

    parameters : list of `float`
        The free parameters to solve for. These are referenced in 
        `parameter_names`.

    observed_spectra : list of `Spectrum1D` objects
        The observed spectra.

    callback : function
        A callback to apply after completing the comparison.
    """

    if "jitter" in parameter_names:
        jitter = parameters[parameter_names.index("jitter")]

        if not (1 > jitter > 0):
            return -np.inf
    else:
        jitter = 0


    # Prepare the observed spectra
    observed_spectra = model.observed_spectra(parameters, parameter_names, observed_spectra)
    if observed_spectra is None:
        return -np.inf

    # Get the synthetic spectra
    model_spectra = model.model_spectra(parameters, parameter_names, observed_spectra)
    if model_spectra is None:
        return -np.inf

    # Any masks?
    masks = model.masks(model_spectra)
    weighting_functions = model.weights(model_spectra)
    
    # Calculate chi^2 difference
    chi_sq_i = {}
    for i, (aperture, observed_spectrum) in enumerate(zip(model._mapped_apertures, observed_spectra)):

        chi_sq = (observed_spectrum.flux - model_spectra[aperture].flux)**2/(observed_spectrum.uncertainty**2 + jitter)
        
        # Apply any weighting functions to the chi_sq values
        chi_sq /= weighting_functions[aperture](model_spectra[aperture].disp, model_spectra[aperture].flux)

        # Apply masks
        chi_sq *= masks[aperture]

        # Add only finite, positive values
        positive_finite_chisq_indices = (chi_sq > 0) * np.isfinite(chi_sq)
        positive_finite_flux_indices = (observed_spectrum.flux > 0) * np.isfinite(observed_spectrum.flux)

        # Useful_pixels of 1 indicates that we should use it, 0 indicates it was masked.
        useful_pixels = positive_finite_chisq_indices * positive_finite_flux_indices

        chi_sq_i[aperture] = chi_sq[useful_pixels]

        # Update the masks values:
        #> -2: Not interested in this region, and it was non-finite (not used).
        #> -1: Interested in this region, but it was non-finite (not used).
        #>  0: Not interested in this region, it was finite (not used).
        #>  1: Interested in this region, it was finite (used for \chi^2 determination)
        masks[aperture][~useful_pixels] -= 2

    total_chi_sq = np.sum(map(np.sum, chi_sq_i.values()))
    
    return total_chi_sq

# coding: utf-8

""" Handles the analysis for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

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

__all__ = ['analyze', 'analyze_star', 'prepare_model_spectra', 'prepare_observed_spectra', 'chi_squared_fn']

class CallbackClass(object):
    pass


class Worker(multiprocessing.Process):
    """A SCOPE worker to analyse stellar spectra."""

    def __init__(self, queue_in, queue_out):
        super(Worker, self).__init__()
        self.queue_in = queue_in
        self.queue_out = queue_out

        logging.debug("Initialised a new Worker.")

    def run(self):
        logging.debug("Running new Worker process.")

        for data in iter(self.queue_in.get, None):
            try:
                result = analyze_star(*data)

            except:
                self.queue_out.put(False)
                etype, value, tb = sys.exc_info()
                logging.warn("Failed to analyse spectra:\n\tTraceback (most recent call last):\n{traceback}\n{etype}: {value}"
                    .format(traceback=tb, etype=etype, value=value))

            else:
                self.queue_out.put(result)


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
                    and parameter_name.lower().startswith("doppler_correct.") and parameter_name.lower().endswith(".allow_shift"):

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

                elif prior_value.lower().startswith("gaussian"):
                    mu, sigma = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                    if i == 0:
                        logging.info("Initialised {0} parameter with a gaussian distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
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
            logging.info("\tParameter {0} - initial point: {1.2e}".format(
                ordered_parameter_name, walker_priors[i]))

    return (ordered_parameter_names, walker_priors)


def analyze_all(stars, configuration_filename, output_filename_prefix=None, clobber=False, 
    callback=None, timeout=120):
    """Analyse a number of stars according to the configuration provided.

    Inputs
    ------
    stars : iterable of spectra
        Each star can contain multiple spectra, but they must be in sub-lists.

    configuration_filename : str
        The configuration filename for this analysis.

    output_filename_prefix : str, optional
        The filename prefix to use for saving results. If this is specified, a summary file
        will be produced, and individual results will be saved.

    clobber : bool, default False
        Whether to overwrite existing output files if they already exist. 

    callback : function, optional
        A callback to perform after every model comparison. If you're running
        things in parallel, I hope you know what you're doing.

    timout : int, default is 120
        The number of seconds to wait for each parallel thread to send results
        before killing the thread.
    """

    # Load the configuration
    configuration = config.load(configuration_filename)

    # Check for threading without emcee
    if configuration.get("threads", 1) > 1 and configuration["solution_method"] != "emcee":

        logging.info("Initializing {n} parallel workers".format(n=configuration['threads']))

        # Do parallel
        queue_in = multiprocessing.Queue()
        queue_out = multiprocessing.Queue()

        # Initialise all the workers
        for i in xrange(configuration['threads']):
            Worker(queue_in, queue_out).start()

        for star in stars:
            queue_in.put((star, configuration, callback))

        # Shut down all workers
        [queue_in.put(None) for i in xrange(configuration['threads'])]

        # Get all the results
        results = []
        while len(results) < len(stars):
            try:
                results.append(queue_out.get(timeout=timeout))
            except: continue

    else:

        if solution_method != "emcee":
            logging.info("Performing analysis in serial mode")

        # Do serial
        results = [analyze_star(star, configuration, callback) for star in stars]

    # Should we be saving the results?
    if output_filename_prefix is not None:

        summary_filename = output_filename_prefix + ".csv"


        logging.info("Summarising results to {summary_filename}".format(summary_filename=summary_filename))

        # What columns do we want in our summary file?
        # star ID, pickle filename, ra, dec, object name, ut_date, ut_time, airmass, exposure time?, chi^2, 
        # ndof, r_chi^2, S/N estimates? all the parameters solved for

        summary_lines = []
        sorted_posterior_keys = None
        observed_headers_requested = ["RA", "DEC", "OBJECT", ]
        
        for i, result in enumerate(results, start=1):

            if result in (None, False) or np.isnan(result[0]):
                line_data = ["Star #{i}".format(i=i), ""]
                
                # Add in the observed headers to the line data.
                if result not in (None, False) and len(result) > 3 and result[3] != None:
                    observed_spectra = result[3]
                    line_data += [observed_spectra[0].headers[header] \
                        if header in observed_spectra[0].headers else "" for header in observed_headers_requested]
                    
                else:
                    line_data += [""] * len(observed_headers_requested)

                # Fill the \chi^2, DOF, reduced \chi^2 with blanks
                line_data += ["", "", ""]
                
            else:

                # Create a pickle filename
                pickle_filename = "{prefix}-star-{i}.pickle".format(prefix=output_filename_prefix, i=i)

                # Save results to pickle
                if os.path.exists(pickle_filename) and not clobber:
                    logging.warn("Pickle filename {filename} already exists and we've been asked not to clobber it. Skipping.."
                        .format(filename=pickle_filename))

                else:
                    with open(pickle_filename, "w") as fp:
                        pickle.dump(result, fp, -1)
            
                    logging.info("Results for Star #{i} saved to {filename}".format(i=i, filename=pickle_filename))
        
                line_data = [
                    "Star #{i}".format(i=i),
                    pickle_filename,
                ]

                chi_sq, num_dof, posteriors, observed_spectra, model_spectra, masks = result

                if sorted_posterior_keys is None:
                    sorted_posterior_keys = sorted(posteriors.keys(), key=len)

                # Add in observed headers to line data
                [line_data.append(observed_spectra[0].headers[header]) 
                    if header in observed_spectra[0].headers else "" for header in observed_headers_requested]

                # Add in results information
                line_data.extend([chi_sq, num_dof, chi_sq/num_dof])

                # Add in the posteriors
                line_data.extend([posteriors[key] for key in sorted_posterior_keys])

            # Add this line in
            summary_lines.append(line_data)
            
        column_headers = ["Name", "Results filename"] + observed_headers_requested \
            + ["\chi^2", "DOF", "Reduced \chi^2"] + sorted_posterior_keys

        # Fill in any 'non-results' with the appropriate number of blanks
        summary_lines_formatted = []
        for line in summary_lines:
            if len(line) < len(column_headers):
                line += [""] * (len(column_headers) - len(line))

            summary_lines_formatted.append(line)

        # Get max lengths
        max_lengths = [0] * len(column_headers)
        for line in summary_lines_formatted:
            for i, length in enumerate(map(len, map(str, line))):
                if length + 1 > max_lengths[i]:
                    max_lengths[i] = length + 1

        column_headers = [header.ljust(length) for header, length in zip(column_headers, max_lengths)]
        summary_lines_formatted = [[item.ljust(length) for item, length in zip(line, max_lengths)] for line in summary_lines_formatted]

        # Save the summary file
        if os.path.exists(summary_filename) and not clobber:
            logging.warn("Summary filename {filename} already exists and we've been asked not to clobber it. Logging summary results instead.")

            logging.info(",".join(column_headers))
            logging.info("\n".join([",".join(line) for line in summary_lines_formatted]))

        else:

            with open(summary_filename, "w") as fp:
                fwriter = csv.writer(fp, delimiter=",")

                fwriter.writerow(column_headers)
                fwriter.writerows(summary_lines_formatted)

            logging.info("Summary file saved to {filename}".format(filename=summary_filename))


    return results


def analyze_star(observed_spectra, configuration, lnprob_fn_callback=None, chi_squared_fn_callback=None):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    configuration : dict
        The configuration settings for this analysis.

    chi_squared_fn_callback : function
        A callback to perform after every model comparison.
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
    model = models.Models(configuration)

    # Get the aperture mapping from observed spectra to model spectra
    # For example, which index in our list of spectra corresponds to
    # 'blue', or 'red' in our model 
    aperture_mapping = model.map_apertures(observed_dispersions)

    # Check that the mean pixel size in the model dispersion maps is smaller than the observed dispersion maps
    for aperture, observed_dispersion in zip(aperture_mapping, observed_dispersions):

        mean_observed_pixel_size = np.mean(np.diff(observed_dispersion))
        mean_model_pixel_size = np.mean(np.diff(model.dispersion[aperture]))

        if mean_model_pixel_size > mean_observed_pixel_size:
            raise ValueError("the mean model pixel size in the {aperture} aperture is larger than the mean"
                " pixel size in the observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                .format(
                    aperture=aperture,
                    wl_start=np.min(observed_dispersion),
                    wl_end=np.max(observed_dispersion)))
    
    # Make fmin_powell the default
    if "solution_method" not in configuration or configuration["solution_method"] == "fmin_powell":

        fail_value = 999
        parameter_names, p0 = initialise_priors(configuration, model, observed_spectra, aperture_mapping)
        optimisation_args = (parameter_names, observed_spectra, aperture_mapping, model, configuration, fail_value, chi_squared_fn_callback)   
        parameters_final = scipy.optimize.fmin_powell(chi_squared_fn, p0, args=optimisation_args, xtol=0.001, ftol=0.001)

    elif configuration["solution_method"] == "leastsq":

        fail_value = 999
        parameter_names, p0 = initialise_priors(configuration, model, observed_spectra, aperture_mapping)
        optimisation_args = (parameter_names, observed_spectra, aperture_mapping, model, configuration, fail_value, chi_squared_fn_callback)
        parameters_final = scipy.optimize.leastsq(chi_squared_fn, p0, args=optimisation_args, xtol=1e-3, ftol=1e-3)

    elif configuration["solution_method"] == "emcee":

        # Ensure we have the number of walkers and steps specified in the configuration
        nwalkers, nsteps = configuration["emcee"]["nwalkers"], configuration["emcee"]["nsteps"]
        lnprob0, rstate0 = None, None
        mean_acceptance_fractions = np.zeros(nsteps)
        
        # Initialise priors and set up arguments for optimization
        parameter_names, p0 = initialise_priors(configuration, model, observed_spectra, aperture_mapping, nwalkers)
        optimisation_args = (parameter_names, observed_spectra, aperture_mapping, model, configuration, 
            lnprob_fn_callback, chi_squared_fn_callback)   

        logging.info("All priors initialsed for {0} walkers. Parameter names are: {1}".format(nwalkers, ", ".join(parameter_names)))
        sampler = emcee.EnsembleSampler(nwalkers, len(parameter_names), lnprob_fn, args=optimisation_args,
            threads=configuration.get("threads", 1))

        # Sample_data contains all the inputs, and the \chi^2 and L 
        for i, (pos, lnprob, state, blobs) in enumerate(sampler.sample(
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


def prepare_model_spectra(parameters, parameter_names, observed_spectra, aperture_mapping, model, configuration):
    """Interpolates the flux for a set of stellar parameters and prepares the model spectra
    for comparison. This includes any smoothing, resampling, and normalisation of the data.

    Inputs
    ------
    parameters : list of floats
        The input parameters that were provdided to the `chi_squared_fn` function.

    parameter_names : list of str, should be same length as `parameters`.
        The names for the input parameters.

    observed_spectra : list of `Spectrum1D` objects
        The observed spectra.

    aperture_mapping : list of `str`, same length as `observed_spectra`
        The names of the model apertures associated to each observed spectrum.

    model : `models.Model` class
        The model class containing the reference to the grid of model atmospheres.

    configuration : `dict`
        The configuration class.
    """

    # Build the grid point
    stellar_parameters = model.colnames
    grid_point = [parameters[parameter_names.index(stellar_parameter)] for stellar_parameter in stellar_parameters]

    # Get interpolated flux
    try:
        synthetic_fluxes = model.interpolate_flux(grid_point)

    except:
        logging.debug("No model flux could be determined for {0}".format(
            ", ".join(["{0} = {1:.2f}".format(parameter, value) for parameter, value in zip(stellar_parameters, grid_point)])
            ))
        return None

    logging.debug("Interpolated model flux at {0}".format(
        ", ".join(["{0} = {1:.2f}".format(parameter, value) for parameter, value in zip(stellar_parameters, grid_point)])
        ))

    if synthetic_fluxes == {}: return None
    for aperture, flux in synthetic_fluxes.iteritems():
        if np.all(~np.isfinite(flux)): return None

    # Create spectra
    model_spectra = {}
    for aperture, synthetic_flux in synthetic_fluxes.iteritems():
        model_spectra[aperture] = specutils.Spectrum1D(
                                          disp=model.dispersion[aperture],
                                          flux=synthetic_flux)

    # Any synthetic smoothing to apply?
    for aperture in aperture_mapping:
        key = 'smooth_model_flux.{aperture}.kernel'.format(aperture=aperture)

        # Is the smoothing a free parameter?
        if key in parameter_names:
            index = parameter_names.index(key)
            # Ensure valid smoothing value
            if parameters[index] < 0: return
            model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(parameters[index])

        elif configuration['smooth_model_flux'][aperture]['perform']:
            # Ensure valid smoothing value
            if configuration['smooth_model_flux'][aperture]['kernel'] < 0: return

            # It's a fixed value.
            model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(configuration['smooth_model_flux'][aperture]['kernel'])
            logging.debug("Smoothed model flux for '{aperture}' aperture".format(aperture=aperture))

    # Perform normalisation if necessary
    if "normalise_model" in configuration:
        for aperture, observed_spectrum in zip(aperture_mapping, observed_spectra):
            if  aperture in configuration["normalise_model"] \
            and configuration["normalise_model"][aperture]["perform"]:
                
                # Perform normalisation here
                normalisation_kwargs = {}
                normalisation_kwargs.update(configuration["normalise_model"][aperture])

                # Now update these keywords with priors
                for parameter_name, parameter in zip(parameter_names, parameters):
                    if parameter_name.startswith("normalise_model.{aperture}.".format(aperture=aperture)):

                        parameter_name_sliced = '.'.join(parameter_name.split('.')[2:])
                        normalisation_kwargs[parameter_name_sliced] = parameter

                # Normalise the spectrum
                logging.debug("Normalisation arguments for model '{aperture}' aperture: {kwargs}"
                    .format(aperture=aperture, kwargs=normalisation_kwargs))

                try:
                    normalised_spectrum, continuum = model_spectra[aperture].fit_continuum(**normalisation_kwargs)

                except:
                    logging.debug("Normalisation of model spectra in {0} aperture failed".format(aperture))
                    return None

                else:
                    model_spectra[aperture] = normalised_spectrum

    # Interpolate synthetic to observed dispersion map
    for aperture, observed_spectrum in zip(aperture_mapping, observed_spectra):
        model_spectra[aperture] = model_spectra[aperture].interpolate(observed_spectrum.disp)

    return model_spectra


def prepare_observed_spectra(parameters, parameter_names, observed_spectra, aperture_mapping, configuration):
    """Prepares the observed spectra for comparison against model spectra by performing
    normalisation and doppler shifts to the spectra.

    Inputs
    ------
    parameters : list of floats
        The input parameters that were provdided to the `chi_squared_fn` function.

    parameter_names : list of str, should be same length as `parameters`.
        The names for the input parameters.

    observed_spectra : list of `Spectrum1D` objects
        The observed spectra.

    aperture_mapping : list of `str`, same length as `observed_spectra`
        The names of the model apertures associated to each observed spectrum.

    configuration : `dict`
        The configuration class.
    """

    logging.debug("Preparing observed spectra for comparison")

    # Any normalisation to perform?
    normalised_spectra = []
    for aperture, spectrum in zip(aperture_mapping, observed_spectra):
        if not configuration['normalise_observed'][aperture]['perform']:
            normalised_spectra.append(spectrum)
            continue

        normalisation_kwargs = {}
        normalisation_kwargs.update(configuration['normalise_observed'][aperture])

        # Now update these keywords with priors
        for parameter_name, parameter in zip(parameter_names, parameters):
            if parameter_name.startswith('normalise_observed.{aperture}.'.format(aperture=aperture)):

                parameter_name_sliced = '.'.join(parameter_name.split('.')[2:])
                normalisation_kwargs[parameter_name_sliced] = parameter

        # Normalise the spectrum
        logging.debug("Normalisation arguments for '{aperture}' aperture: {kwargs}"
            .format(aperture=aperture, kwargs=normalisation_kwargs))

        try:
            normalised_spectrum, continuum = spectrum.fit_continuum(**normalisation_kwargs)

        except:
            return None

        else:

            if spectrum.uncertainty is None:
                normalised_spectrum.uncertainty = continuum.flux**(-0.5)
            
            normalised_spectra.append(normalised_spectrum)

        logging.debug("Performed normalisation for aperture '{aperture}'".format(aperture=aperture))


    # Any doppler shift?
    for i, aperture in enumerate(aperture_mapping):
        key = 'doppler_correct.{aperture}.allow_shift'.format(aperture=aperture)

        if key in parameter_names:
            index = parameter_names.index(key)
            normalised_spectra[i] = normalised_spectra[i].doppler_shift(parameters[index])

            logging.debug("Performed doppler shift of {velocity:.2f} km/s for aperture '{aperture}'"
                .format(aperture=aperture, velocity=parameters[index]))

    return normalised_spectra


def prepare_weights(model_spectra, configuration):
    """Returns callable weighting functions to apply to the \chi^2 comparison.

    Inputs
    ------
    model_spectra : dict
        A dictionary containing aperture names as keys and specutils.Spectrum1D objects
        as values.

    configuration_filename : dict
        The configuration dictionary.
    """

    if "weights" not in configuration:
        weights = {}
        for aperture, spectrum in model_spectra.iteritems():
            # Equal weighting to all pixels
            weights[aperture] = lambda disp, flux: np.ones(len(flux))

    else:
        weights = {}
        for aperture, spectrum in model_spectra.iteritems():
            if aperture not in configuration["weights"]:
                # Equal weighting to all pixels
                weights[aperture] = lambda disp, flux: np.ones(len(flux))

            else:
                # Evaluate the expression, providing numpy (as np), disp, and flux as locals
                weights[aperture] = lambda disp, flux: eval(configuration["weights"][aperture], 
                    {"disp": disp, "np": np, "flux": flux})

    return weights


def prepare_masks(model_spectra, configuration):
    """Returns pixel masks to apply to the model spectra
    based on the configuration provided.

    Inputs
    ------
    model_spectra : dict
        A dictionary containing aperture names as keys and specutils.Spectrum1D objects
        as values.

    configuration : dict
        The configuration dictionary.
    """

    if "masks" not in configuration:
        masks = {}
        for aperture, spectrum in model_spectra.iteritems():
            masks[aperture] = np.ones(len(spectrum.disp))

    else:
        masks = {}
        for aperture, spectrum in model_spectra.iteritems():
            if aperture not in configuration["masks"]:
                masks[aperture] = np.ones(len(spectrum.disp))
            
            else:
                # We are required to build a mask.
                mask = np.zeros(len(spectrum.disp))
                if configuration["masks"][aperture] is not None:
                    for region in configuration["masks"][aperture]:
                        index_start, index_end = np.searchsorted(spectrum.disp, region)
                        mask[index_start:index_end] = 1

                masks[aperture] = mask


    return masks


def chi_squared_fn(parameters, parameter_names, observed_spectra, aperture_mapping, \
    model, configuration, fail_value=999, callback=None):
    """Calculates the \chi^2 difference between observed and
    synthetic spectra.

    parameters : list of `float`
        The free parameters to solve for. These are referenced in 
        `parameter_names`.

    observed_spectra : list of `Spectrum1D` objects
        The observed spectra.

    callback : function
        A callback to apply after completing the comparison.
    """

    assert len(parameters) == len(parameter_names)

    # Prepare the observed spectra
    observed_spectra = prepare_observed_spectra(parameters, parameter_names, observed_spectra, aperture_mapping, configuration)
    if observed_spectra is None:
        return fail_value

    # Get the synthetic spectra
    model_spectra = prepare_model_spectra(parameters, parameter_names, observed_spectra, aperture_mapping, model, configuration)
    if model_spectra is None:
        return fail_value

    # Any masks?
    masks = prepare_masks(model_spectra, configuration)

    # Any weighting functions?
    weighting_functions = prepare_weights(model_spectra, configuration)

    # Calculate chi^2 difference
    chi_sq_i = {}
    for i, (aperture, observed_spectrum) in enumerate(zip(aperture_mapping, observed_spectra)):

        jitter = parameters[parameter_names.index("jitter")] if "jitter" in parameter_names else 0
        chi_sq = (observed_spectrum.flux - model_spectra[aperture].flux)**2/(observed_spectrum.uncertainty**2 + jitter)
        
        # Apply any weighting functions to the chi_sq values
        chi_sq *= weighting_functions[aperture](model_spectra[aperture].disp, model_spectra[aperture].flux)

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

    num_pixels = sum(map(len, chi_sq_i.values()))
    total_chi_sq = np.sum(map(np.sum, chi_sq_i.values()))
    
    num_dof = num_pixels - len(parameters) - 1

    # Return likelihood
    logging.debug((parameters, total_chi_sq, num_dof, total_chi_sq/num_dof))

    if callback is not None:
        # Perform the callback function
        callback(
            total_chi_sq,
            num_dof,
            OrderedDict(zip(parameter_names, parameters)),
            observed_spectra,
            [model_spectra[aperture] for aperture in aperture_mapping],
            [masks[aperture] for aperture in aperture_mapping]
            )

    logging.debug("Total \chi^2: {chi_sq}, n_dof: {ndof}".format(chi_sq=total_chi_sq, ndof=num_dof))
    return total_chi_sq


def lnprob_fn(parameters, parameter_names, observed_spectra, aperture_mapping, \
    model, configuration, lnprob_callback=None, chi_squared_fn_callback=None, **kwargs):
    """Calculates the log of the probability of the $\chi^2$ distribution function.

    Inputs the same as `chi_squared_fn_fn`"""

    # Check the jitter
    jitter = parameters[parameter_names.index("jitter")]
    if not 0 <= jitter <= 1:

        blobs = list(parameters) + [np.inf, -np.inf]
        return (-np.inf, blobs)

    # Need uncertainty for the pixels that we're actually sampling!
    # We're going to piggy-back on any existing callback
    pixel_uncertainty = []
    def callback(*args):

        observed_spectra, masks = args[-3], args[-1]
        for observed_spectrum, mask in zip(observed_spectra, masks):
            indices = np.where(mask == 1)[0]
            pixel_uncertainty.extend(observed_spectrum.uncertainty[indices])

        if chi_squared_fn_callback is not None:
            chi_squared_fn_callback(*args)

    chi_sq = chi_squared_fn(parameters, parameter_names, observed_spectra, aperture_mapping,
        model, configuration, fail_value=np.inf, callback=callback)

    pixel_uncertainty = np.array(pixel_uncertainty)
    inverse_variance = 1 / (pixel_uncertainty**2 + jitter)
    log_likelihood = -0.5 * (chi_sq - np.sum(np.log(inverse_variance)))
    
    logging.debug("lnprob_fn({0}), \n\tyields $\chi^2$ = {1:.2f}, log-likelihood: {2:.5e}".format(
        ", ".join(["{0}: {1:.2f}".format(p, v) for p, v in zip(parameter_names, parameters)]),
        chi_sq, log_likelihood))

    if lnprob_callback is not None:
        lnprob_callback(parameters, chi_sq, log_likelihood)

    blobs = list(parameters) + [chi_sq, log_likelihood]
    return (log_likelihood, blobs)

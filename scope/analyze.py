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
import time

# Third-party
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import config
import models
import utils
import specutils

__all__ = ['analyze', 'analyze_star', 'prepare_model_spectra', 'prepare_observed_spectra', 'chi_squared']

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

            else:
                self.queue_out.put(result)


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

    # Check for parallelism
    if 'parallelism' in configuration and configuration['parallelism'] > 1:

        logging.info("Initializing {n} parallel workers".format(n=configuration['parallelism']))

        # Do parallel
        queue_in = multiprocessing.Queue()
        queue_out = multiprocessing.Queue()

        # Initialise all the workers
        for i in xrange(configuration['parallelism']):
            Worker(queue_in, queue_out).start()

        for star in stars:
            queue_in.put((star, configuration, callback))

        # Shut down all workers
        [queue_in.put(None) for i in xrange(configuration['parallelism'])]

        # Get all the results
        results = []
        while len(results) < len(stars):
            try:
                results.append(queue_out.get(timeout=timeout))
            except: continue

    else:

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
        observed_headers_requested = ["RA", "DEC", "OBJECT", "AIRMASS", "EXPTIME"]
        
        for i, result in enumerate(results, start=1):

            if result is None:
                line_data = ["Star #{i}".format(i=i), ""]
                
                # Add in the observed headers to the line data.
                #line_data += [observed_spectra[0].headers[header] \
                #    if header in observed_spectra[0].headers else "" for header in observed_headers_requested]
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
                        pickle.save(result)
            
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


def analyze_star(observed_spectra, configuration, callback=None):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    configuration : dict
        The configuration settings for this analysis.

    callback : function
        A callback to perform after every model comparison.
    """

    if 0 in [len(spectrum.disp) for spectrum in observed_spectra]: return None

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

    # Do we have uncertainties in our spectra? If not we should estimate it.
    for spectrum in observed_spectra:
        if spectrum.uncertainty is None:
            spectrum.uncertainty = spectrum.flux**(-0.5)

    # Initialise priors
    parameter_names = []
    parameters_initial = []
    
    for parameter_name, parameter in configuration['priors'].iteritems():
        parameter_names.append(parameter_name)

        try:
            float(parameter)

        except:
            # We probably need to evaluate this.
            if parameter == "uniform":
                # Only works on stellar parameter values.

                index = model.colnames.index(parameter_name)
                possible_points = model.grid_points[:, index]

                parameters_initial.append(random.uniform(np.min(possible_points), np.max(possible_points)))

            else:
                raise TypeError("prior type not valid for {parameter_name}".format(parameter_name=parameter_name))

        else:
            parameters_initial.append(parameter)

    # Measure the radial velocity if required
    velocities = {}
    apertures_without_measurements = []
    for aperture, spectrum in zip(aperture_mapping, observed_spectra):

        # Should we be measuring velocity for this aperture?
        if 'measure' in configuration['doppler_correct'][aperture] \
        and configuration['doppler_correct'][aperture]['measure']:
            velocity, velocity_err = spectrum.cross_correlate(
                specutils.Spectrum1D.load(configuration['doppler_correct'][aperture]['template']),
                configuration['doppler_correct'][aperture]['wavelength_region']
                )

            velocities[aperture] = velocity
            logging.debug("Measured velocity in {aperture} aperture to be {velocity:.2f} km/s"
                .format(aperture=aperture, velocity=velocity))

            if configuration['doppler_correct'][aperture]['allow_shift'] and abs(velocity) < 500.:
                logging.debug("Updating prior 'doppler_correct.{aperture}.allow_shift' with measured velocity {velocity:.2f} km/s"
                    .format(aperture=aperture, velocity=-velocity))

                parameter_name = 'doppler_correct.{aperture}.allow_shift'.format(aperture=aperture)
                parameters_initial[parameter_names.index(parameter_name)] = -velocity
                configuration['priors'][parameter_name] = -velocity

            elif abs(velocity) > 500.:
                logging.warn("Ignoring velocity measurement for prior 'doppler_correct.{aperture}.allow_shift' because it was too "
                    "high: {velocity:.2f} km/s".format(aperture=aperture, velocity=velocity))

        elif configuration['doppler_correct'][aperture]['allow_shift']:
            apertures_without_measurements.append(aperture)

    if len(velocities) > 0:
        mean_velocity = np.mean(velocities.values())

        if abs(mean_velocity) > 500.:
            logging.warn("Ignoring mean velocity measurement for other apertures because it was too high: {mean_velocity:.2f} km/s"
                .format(mean_velocity=mean_velocity))

        else:

            for aperture in apertures_without_measurements:
                logging.debug("Updating prior 'doppler_correct.{aperture}.allow_shift' with mean velocity {mean_velocity:.2f} km/s"
                    .format(aperture=aperture, mean_velocity=-mean_velocity))

                parameter_name = 'doppler_correct.{aperture}.allow_shift'.format(aperture=aperture)
                parameters_initial[parameter_names.index(parameter_name)] = -mean_velocity
                configuration['priors'][parameter_name] = -mean_velocity

    elif len(velocities) > 0:
        logging.warn("There are apertures that allow a velocity shift but no mean velocity could be determined"
            " from other apertures.")

    fail_value = 999
    optimisation_args = (parameter_names, observed_spectra, aperture_mapping, model, configuration, fail_value, callback)
    
    parameters_final = scipy.optimize.fmin_powell(chi_squared, parameters_initial, args=optimisation_args, xtol=0.01, ftol=0.01)

    # We will need to sample the chi_squared function again with a callback to save
    # the results
    output = CallbackClass()
    final_callback = lambda *x: setattr(output, 'data', x)
    chi_squared(parameters_final, parameter_names, observed_spectra, aperture_mapping, model, configuration, fail_value, final_callback)
    
    try:
        chi_sq, num_dof, posteriors, observed_spectra, model_spectra, masks = output.data

    except AttributeError:
        return (np.nan, np.nan, None, None, None, None)

    else:
        return (chi_sq, num_dof, posteriors, observed_spectra, model_spectra, masks)


def prepare_model_spectra(parameters, parameter_names, observed_spectra, aperture_mapping, model, configuration):
    """Interpolates the flux for a set of stellar parameters and prepares the model spectra
    for comparison (i.e. smoothing and resampling).

    Inputs
    ------
    parameters : list of floats
        The input parameters that were provdided to the `chi_squared` function.

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

    except: return None

    logging.debug("Interpolated model flux")

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
            model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(parameters[index])

        elif configuration['smooth_model_flux'][aperture]['perform']:
            # It's a fixed value.
            model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(configuration['smooth_model_flux'][aperture]['kernel'])
            logging.debug("Smoothed model flux for '{aperture}' aperture".format(aperture=aperture))

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
        The input parameters that were provdided to the `chi_squared` function.

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
                for region in configuration["masks"][aperture]:
                    index_start, index_end = np.searchsorted(spectrum.disp, region)

                    mask[index_start:index_end] = 1

                masks[aperture] = mask

    return masks


def chi_squared(parameters, parameter_names, observed_spectra, aperture_mapping, \
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
    if observed_spectra is None: return fail_value

    # Get the synthetic spectra
    model_spectra = prepare_model_spectra(parameters, parameter_names, observed_spectra, aperture_mapping, model, configuration)
    if model_spectra is None: return fail_value

    # Any masks?
    masks = prepare_masks(model_spectra, configuration)

    # Any weighting functions?
    #weights = prepare_weights(model_spectra, configuration)

    # Calculate chi^2 difference
    chi_sq_i = {}
    for i, (aperture, observed_spectrum) in enumerate(zip(aperture_mapping, observed_spectra)):

        chi_sq = ((observed_spectrum.flux - model_spectra[aperture].flux)**2)/observed_spectrum.uncertainty

        # Apply masks
        chi_sq *= masks[aperture]

        # Pearson's \chi^2:
        #chi_sq = ((observed_spectrum.flux - model_spectra[aperture].flux)**2)/model_spectra[aperture].flux

        # Add only finite, positive values
        finite_chisq_indices = np.isfinite(chi_sq)
        positive_finite_flux_indices = (observed_spectrum.flux > 0) * np.isfinite(observed_spectrum.flux)

        # useful_pixels of 1 indicates that we should use it, 0 indicates it was masked.
        useful_pixels = finite_chisq_indices * positive_finite_flux_indices

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
            dict(zip(parameter_names, parameters)),
            observed_spectra,
            [model_spectra[aperture] for aperture in aperture_mapping],
            [masks[aperture] for aperture in aperture_mapping]
            )

    logging.debug("Total \chi^2: {chi_sq}, n_dof: {ndof}".format(chi_sq=total_chi_sq, ndof=num_dof))
    return total_chi_sq

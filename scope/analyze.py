# coding: utf-8

""" Handles the analysis for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os

# Third-party
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import config
import models
import utils
import specutils


def analyze(observed_spectra, configuration_filename, callback=None):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    configuration_filename : str
        The configuration filename for this analysis.

    callback : function
        A callback to perform after every model comparison.
    """

    # Check observed arms do not overlap
    observed_dispersions = [spectrum.disp for spectrum in observed_spectra]
    overlap = utils.find_spectral_overlap(observed_dispersions)
    if overlap is not None:
        raise ValueError("observed apertures cannot overlap in wavelength, but they do near {wavelength} Angstroms"
            .format(wavelength=overlap))

    # Load the configuration
    configuration = config.load(configuration_filename)

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

    fail_value = 999
    optimisation_args = (parameter_names, observed_spectra, aperture_mapping, model, configuration, fail_value, callback)

    # Get aperture mapping
    #chi_squared(parameters_initial, parameter_names, spectra, aperture_mapping, models, configuration)
    print(parameter_names)
    parameters_final = scipy.optimize.fmin_powell(chi_squared, parameters_initial, args=optimisation_args, xtol=0.01, ftol=0.01)

    results = {}
    for parameter_name, parameter_final in zip(parameter_names, parameters_final):
        results[parameter_name] = parameter_final

    return results



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

    return normalised_spectra


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

    logging.debug("model_spectra")
    logging.debug(model_spectra['blue'].flux)

    # Calculate chi^2 difference
    chi_sq_i = {}
    for i, (aperture, observed_spectrum) in enumerate(zip(aperture_mapping, observed_spectra)):

        chi_sq = ((observed_spectrum.flux - model_spectra[aperture].flux)**2)/observed_spectrum.uncertainty

        # Pearson's \chi^2:
        #chi_sq = ((observed_spectrum.flux - model_spectra[aperture].flux)**2)/model_spectra[aperture].flux

        # Add only finite values
        finite_indices = np.isfinite(chi_sq)
        chi_sq_i[aperture] = chi_sq[finite_indices]

    num_pixels = sum(map(len, chi_sq_i.values()))
    total_chi_sq = np.sum(map(np.sum, chi_sq_i.values()))

    num_dof = num_pixels - len(parameters) - 1
    # Any masks?

    # Return likelihood
    logging.debug((parameters, total_chi_sq, num_dof, total_chi_sq/num_dof))

    if callback is not None:
        # Perform the callback function
        callback(
            total_chi_sq,
            num_dof,
            dict(zip(parameter_names, parameters)),
            observed_spectra,
            [model_spectra[aperture] for aperture in aperture_mapping]
            )

    logging.debug("total chi^2: {chi_sq}, ndof: {ndof}".format(chi_sq=total_chi_sq, ndof=num_dof))
    return total_chi_sq
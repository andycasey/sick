# coding: utf-8

""" Handles the analysis for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os

# Third-party
import numpy as np

# Module
import config
import models
import utils

def analyse(observed_spectra, configuration_filename):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    configuration_filename : str
        The configuration filename for this analysis.
    """

    # Check observed arms do not overlap
    observed_dispersions = [spectrum.disp for spectrum in observed_spectra]
    overlap = utils.find_spectral_overlap(observed_dispersions)
    if overlap is not None:
        raise ValueError("observed apertures cannot overlap in wavelength, but they do near {wavelength} Angstroms"
            .format(wavelength=overlap))

    # Load the configuration
    configuration = config.load(configuration_filename)

    # Load our models
    models = models.Models(configuration)

    # Get the aperture mapping from observed spectra to model spectra
    aperture_mapping = models.map_apertures(observed_dispersions)

    # Check that the mean pixel size in the model dispersion maps is smaller than the observed dispersion maps
    for aperture, observed_dispersion in zip(aperture_mapping, observed_dispersions):

        mean_observed_pixel_size = np.mean(np.diff(observed_dispersion))
        mean_model_pixel_size = np.mean(np.diff(models.dispersion[aperture]))

        if mean_model_pixel_size > mean_observed_pixel_size:
            raise ValueError("the mean model pixel size in the {aperture} aperture is larger than the mean"
                " pixel size in the observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                .format(
                    aperture=aperture,
                    wl_start=np.min(observed_dispersion),
                    wl_end=np.max(observed_dispersion)))

    # Initialise priors
    parameters = []
    parameter_names = []

    optimisation_args = (parameter_names, observed_spectra, named_aperture_mapping, models, configuration)
    # Get aperture mapping
    #chi_squared(parameters, parameter_names, spectra, aperture_mapping, models, configuration)







def log_likelihood(parameters, parameter_names, spectra, aperture_mapping,
    models, configuration, **kwargs):
    """Calculates the log likelihood that a model fits the data."""

    raise NotImplementedError


def chi_squared(parameters, parameter_names, observed_spectra, aperture_mapping, \
    models, configuration):
    """Calculates the \chi^2 difference between observed and
    synthetic spectra.

    parameters : list of `float`
        The free parameters to solve for. These are referenced in 
        `parameter_names`.

    observed_spectra : list of `Spectrum1D` objects
        The observed spectra.
    """


    assert len(parameters) == len(parameter_names)

    # Any normalisation to perform?
    if 'normalise_observed' in configuration['priors']:
        for beam in configuration['priors']['normalise_observed']:

            beam_index = aperture_mapping[beam]

            normalisation_kwargs = {}
            normalisation_kwargs.update(configuration['normalise_observed'][beam])

            # Now update any of those from priors
            normalisation_parameters = {}
            for parameter_name, parameter_value in zip(parameter_names, parameters):
                if parameter_name.startswith('normalised_observed.{beam}.'.format(beam=beam)):
                    normalisation_parameters[parameter_name.split('.')[2]] = parameter_value

            normalisation_kwargs.update(normalisation_parameters)

            spectra[beam_index] = spectra[beam_index].normalise(**normalisation_kwargs)

    # Any doppler shift?

    # Get interpolated flux
    # Build grid_point
    grid_point = []
    #models.grid_points.dtypes

    synthetic_spectra = models.interpolate_flux(grid_point)

    # Any synthetic smoothing?
    if 'smooth_model_flux' in configuration['priors']:
        for beam in configuration['priors']['smooth_model_flux']:

            index = parameter_names.index('smooth_model_flux.{beam}.kernel'.format(beam=beam))
            kernel = parameters[index]

            synthetic_spectra[beam] = synthetic_spectra[beam].gaussian_smooth(kernel)

    # Interpolate synthetic to observed dispersion map
    for beam in synthetic_spectra.keys():
        beam_index = aperture_mapping[beam]

        synthetic_spectra[beam] = synthetic_spectra[beam].interpolate(spectra[beam_index].disp)

    # Calculate chi^2 difference

    # Any masks?

    # Return likelihood
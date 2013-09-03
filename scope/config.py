# coding: utf-8

""" Handles the loading and verification of configuration files for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os

from random import choice

# Third-party
import numpy as np
import yaml

# Module specific
import models

def load(configuration_filename):
    """Loads a JSON-style configuration filename.

    Inputs
    ------
    `configuration_filename` : str
        The configuration filename to load.
    """

    if not os.path.exists(configuration_filename):
        raise IOError("no configuration filename '{filename}' exists".format(filename=configuration_filename))

    with open(configuration_filename, 'r') as fp:
        configuration = yaml.load(fp)

    return configuration


def verify(configuration):
    """Verifies a configuration.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

    # Check optional parameters
    integer_types = ('parallelism', )
    for key in integer_types:
        if key in configuration:
            if not isinstance(configuration[key], (int, )):
                raise TypeError("configuration setting '{key}' is expected "
                        "to be an integer-type".format(key=key))

    # Check the models
    verify_models(configuration)

    # Check the normalisation
    normalisation_priors = verify_normalisation(configuration)

    # Check the smoothing
    smoothing_priors = verify_smoothing(configuration)

    # Check the doppler corrections
    doppler_priors = verify_doppler(configuration)

    # Establish all of the priors    
    priors_to_expect = doppler_priors + smoothing_priors + normalisation_priors

    # Verify that we have priors established for all the priors
    # we expect, and the stellar parameters we plan to solve for
    verify_priors(configuration, priors_to_expect)

    return True


def verify_doppler(configuration):
    """Verifies the doppler component of a configuration.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

    apertures = get_aperture_names(configuration)
    check_aperture_names(configuration, 'doppler_correct')

    if not 'perform_initial_measurement' in configuration['doppler_correct']:
        raise KeyError("configuration setting 'doppler_correct.{aperture}.perform_initial_measurement' not found"
                .format(aperture=aperture))

    if configuration['doppler_correct']['perform_initial_measurement']:

        if not 'initial_template' in configuration['doppler_correct']:
            raise KeyError("configuration setting 'doppler_correct.initial_template' not found")

        elif not os.path.exists(configuration['doppler_correct']['initial_template']):
            raise ValueError("initial template for doppler correct (doppler_correct.initial_template = {filename})"
                " does not exist".format(filename=configuration['doppler_correct']['initial_template']))

    priors_to_expect = []
    for aperture in apertures:
        if 'allow_shift' not in configuration['doppler_correct'][aperture]:
            raise KeyError("configuration setting 'doppler_correct.{aperture}.allow_shift' not found"
                .format(aperture=aperture))

        if configuration['doppler_correct'][aperture]['allow_shift']:
            priors_to_expect.append('doppler_correct.{aperture}.allow_shift'.format(aperture=aperture))

    return priors_to_expect


def verify_priors(configuration, expected_priors):
    """Verifies that the priors in a configuration are valid."""

    # Create a toy model. What parameters (from the model file names) are we solving for?
    toy_model = models.Models(configuration)
    parameters_to_solve = toy_model.colnames

    # Do priors for these values exist?
    for parameter in parameters_to_solve:
        if parameter not in configuration['priors']:
            raise KeyError("no prior found for '{parameter}' parameter".format(parameter=parameter))

    # Verify that these priors make sense.

    # What else can be a prior?
    for expected_prior in expected_priors:
        if expected_prior not in configuration['priors']:
            raise KeyError("no prior found for '{expected_prior}' parameter".format(expected_prior=expected_prior))

    # Do the values for the priors make sense?

    return True


def verify_smoothing(configuration):
    """Verifies the synthetic smoothing component of a configuration.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

    apertures = get_aperture_names(configuration)
    check_aperture_names(configuration, 'smooth_model_flux')

    required_aperture_smooth_settings = ('mode', 'kernel')

    # Verify the settings for each aperture
    priors_to_expect = []
    for aperture in apertures:

        aperture_smooth_settings = configuration['smooth_model_flux'][aperture]

        # Check that settings exist
        if 'perform' not in aperture_smooth_settings:
            raise KeyError("configuration setting 'smooth_model_flux.{aperture}.perform' not found"
                .format(aperture=aperture))

        else:
            for required_setting in required_aperture_smooth_settings:
                if required_setting not in aperture_smooth_settings:
                    raise KeyError("configuration setting 'smooth_model_flux.{aperture}.{required_setting}' not found"
                        .format(aperture=aperture, required_setting=required_setting))

        # Check the types
        for key, value in aperture_smooth_settings.iteritems():
            if key == 'perform' and not value:
                # No smoothing necessary for this aperture
                break

            # Is this going to be a free parameter?
            if value == 'free':
                priors_to_expect.append('smooth_model_flux.{aperture}.{key}'
                    .format(aperture=aperture, key=key))
                continue

            available_modes = ('gaussian', )
            if key == 'mode' and value not in available_modes:
                raise ValueError("configuration setting 'smooth_model_flux.{aperture}.mode' is not valid. Available"
                    " options are: {available_modes}".format(aperture=aperture, available_modes=', '.join(available_modes)))

            if key == 'kernel' and not isinstance(value, (int, float)):
                raise TypeError("configuration setting 'smooth_model_flux.{aperture}.kernel' is expected to be a float-type"
                    .format(aperture=aperture))

    return priors_to_expect


def verify_normalisation(configuration):
    """Verifies the normalisation component of a configuration.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

    apertures = get_aperture_names(configuration)
    check_aperture_names(configuration, 'normalise_observed')
    
    required_aperture_normalisation_settings = ('order', 'knot_spacing', 'upper_clip', 'lower_clip')

    # Verify the settings for each aperture.
    priors_to_expect = []
    for aperture in apertures:

        aperture_normalisation_settings = configuration['normalise_observed'][aperture]

        # Check that settings exist
        if 'perform' not in aperture_normalisation_settings:
            raise KeyError("configuration setting 'normalise_observed.{aperture}.perform' not found"
                .format(aperture=aperture))

        else:
            for required_setting in required_aperture_normalisation_settings:
                if required_setting not in aperture_normalisation_settings:
                    raise KeyError("configuration setting 'normalise_observed.{aperture}.{required_setting}' not found"
                        .format(aperture=aperture, required_setting=required_setting))

        # Check setting value types
        for key, value in aperture_normalisation_settings.iteritems():
            if key == 'perform' and not value:
                # No normalisation necessary for this aperture
                break

            # Is this going to be a free parameter?
            if value == 'free':
                priors_to_expect.append('normalise_observed.{aperture}.{key}'
                    .format(aperture=aperture, key=key))
                continue

            if key in ('order', ):
                if not isinstance(value, (int, )):
                    raise TypeError("configuration setting 'normalise_observed.{aperture}.{key}' is expected "
                        "to be an integer-type".format(aperture=aperture, key=key))

            elif key in ('knot_spacing', 'lower_clip', 'upper_clip'):
                if not isinstance(value, (int, )):
                    raise TypeError("configuration setting 'normalise_observed.{aperture}.{key}' is expected "
                        "to be a float-type".format(aperture=aperture, key=key))

    return priors_to_expect


def get_aperture_names(configuration):
    """Returns the aperture names specified in the configuration."""

    return configuration['models']['dispersion_filenames'].keys()


def check_aperture_names(configuration, key):
    """Checks that all the apertures specified in the models
    exist in the given sub-config `key`.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.

    `key` : str
        The sub-key to check (e.g. 'normalise_observed')
    """

    apertures = get_aperture_names(configuration)
    sub_configuration = configuration[key]

    for aperture in apertures:
        if aperture not in sub_configuration:
            raise KeyError("no aperture '{aperture}' listed in {key}, but"
            " it's specified in the models".format(aperture=aperture, key=key))
    return True


def verify_models(configuration):
    """Verifies the model component of a given configuration.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

    if 'models' not in configuration.keys():
        raise KeyError("no 'models' found in configuration")

    model_configuration = configuration['models']

    # Check the aperture names
    aperture_names = get_aperture_names(configuration)
    protected_aperture_names = ('perform_initial_measurement', 'initial_template')
    
    for aperture in aperture_names:
        if aperture in protected_aperture_names:
            raise ValueError("aperture name '{aperture}' is protected and cannot be used"
                .format(aperture=aperture))

        if '.' in aperture:
            raise ValueError("aperture name '{aperture}' is invalid because aperture names cannot contain a full-stop character '.'"
                .format(aperture=aperture))



    required_keys = ('dispersion_filenames', 'flux_filenames')
    for key in required_keys:
        if key not in model_configuration.keys():
            raise KeyError("no '{key}' found in configuration for models".format(key=key))

    # Check keys exist in both
    missing_keys = [x for x in model_configuration['dispersion_filenames'].keys() if x not in model_configuration['flux_filenames']]

    if len(missing_keys) > 0:
        raise KeyError("missing flux filenames for models in configuration")

    # Check dispersion filenames exist
    dispersion_maps = {}
    for key, dispersion_filename in model_configuration['dispersion_filenames'].iteritems():
        if not os.path.exists(dispersion_filename):
            raise IOError("dispersion filename for {key} arm does not exist: {filename}"
                .format(key=key, filename=dispersion_filename))

        # Load a dispersion map
        dispersion = models.load_model_data(dispersion_filename)
        dispersion_maps[key] = [len(dispersion), np.min(dispersion), np.max(dispersion)]

    # Ensure dispersion maps of standards don't overlap
    all_min = np.min([item[1] for item in dispersion_maps.values()])
    all_max = np.max([item[2] for item in dispersion_maps.values()])

    interval_tree_resolution = 1
    interval_tree_disp = np.arange(all_min, all_max + interval_tree_resolution, interval_tree_resolution)
    interval_tree_flux = np.zeros(len(interval_tree_disp))

    for key, dispersion_map in dispersion_maps.iteritems():

        N, wlstart, wlend = dispersion_map
        idx = np.searchsorted(interval_tree_disp, [wlstart, wlend + interval_tree_resolution])

        interval_tree_flux[idx[0]:idx[1]] += 1

    if np.max(interval_tree_flux) > 1:
        idx = np.where(interval_tree_flux > 1)[0]
        wavelength = interval_tree_disp[idx[0]]

        raise ValueError("dispersion maps overlap near {wavelength}".format(wavelength=wavelength))

    # Try actually loading a model
    toy_model = models.Models(configuration)

    # Check the length of the dispersion map and some random point
    for aperture, dispersion_map in toy_model.dispersion.iteritems():
        n_dispersion_points = len(dispersion_map)

        random_filename = choice(toy_model.flux_filenames[aperture])
        random_flux = models.load_model_data(random_filename)
        n_flux_points = len(random_flux)

        if n_dispersion_points != n_flux_points:
            raise ValueError("number of dispersion points ({n_dispersion_points}) and flux points ({n_flux_points})"
                " does not match for {aperture} aperture in randomly selected filename: {random_filename}"
                .format(n_dispersion_points=n_dispersion_points, aperture=aperture, n_flux_points=n_flux_points,
                    random_filename=random_filename))

    return True

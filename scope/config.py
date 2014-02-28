# coding: utf-8

""" Handles the loading and verification of configuration files for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import json
import logging
import os
from random import choice

logger = logging.getLogger(__name__)

# Third-party
import numpy as np
try:
    import yaml
except ImportError:
    logger.warn("YAML module not loaded. Only JSON configuration files can be parsed.")

# Module specific
import models

def load(configuration_filename):
    """Loads a configuration filename (either YAML or JSON)

    Inputs
    ------
    `configuration_filename` : str
        The configuration filename to load.
    """

    if not os.path.exists(configuration_filename):
        raise IOError("no configuration filename '{filename}' exists".format(
            filename=configuration_filename))

    module = json if configuration_filename.endswith(".json") else yaml

    with open(configuration_filename, 'r') as fp:
        configuration = module.load(fp)

    return configuration


def verify(configuration):
    """Verifies a SCOPE configuration dictionary that everything makes sense.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

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

    # Verify masks and weights (both are optional)
    verify_masks(configuration)
    verify_weights(configuration)

    # Verify solution setup
    verify_solver(configuration)

    return True


def verify_solver(configuration):
    """ Verifies the configuration settings for which solver to employ. """

    if "solver" not in configuration:
        raise KeyError("no solver information provided in configuration")

    solver = configuration["solver"]
    available_methods = ("fmin_powell", "emcee")

    if solver["method"] not in available_methods:
        raise ValueError("solver method '{0}' is unsupported. Available methods are: {1}".format(
            solver["method"], ", ".join(available_methods)))

    # Non-solver specific options
    if "threads" in solver and not isinstance(solver["threads"], (float, int)):
        raise TypeError("solver.threads must be an integer-like type")

    return True


def verify_doppler(configuration):
    """ Verifies the doppler component of a configuration dictionary.

    Inputs
    ------
    `configuration` : dict
        A dictionary configuration for SCOPE.
    """

    apertures = get_aperture_names(configuration)
    check_aperture_names(configuration, 'doppler_correct')

    priors_to_expect = []
    for aperture in apertures:
        if 'perform' not in configuration['doppler_correct'][aperture]:
            raise KeyError("configuration setting 'doppler_correct.{aperture}.perform' not found"
                .format(aperture=aperture))

        # Radial velocity as a prior?
        if configuration['doppler_correct'][aperture]['perform']:
            priors_to_expect.append('doppler_correct.{aperture}.perform'.format(aperture=aperture))

        # Should we be measuring radial velocity?
        if 'measure' in configuration['doppler_correct'][aperture] \
        and configuration['doppler_correct'][aperture]['measure']:

            # Check the template and wavelength range
            if 'template' not in configuration['doppler_correct'][aperture]:
                raise KeyError("configuration setting 'doppler_correct.{aperture}.template' not found"
                    ", and is needed because doppler_correct.{aperture}.measure is true".format(aperture=aperture))

            if 'wavelength_region' not in configuration['doppler_correct'][aperture]:
                raise KeyError("configuration setting 'doppler_correct.{aperture}.wavelength_region' not found"
                    ", and is needed becasue doppler_correct.{aperture}.measure is true".format(aperture=aperture))

            # Check the template filename exists
            if not os.path.exists(configuration['doppler_correct'][aperture]['template']):
                raise IOError("doppler correct template filename for {aperture} arm does not exist: {filename}"
                    .format(aperture=aperture, filename=configuration['doppler_correct'][aperture]['template']))

    return priors_to_expect


def verify_priors(configuration, expected_priors):
    """ Verifies that the priors in a configuration are valid. 

    Inputs
    ------
    configuration : dict
        A dictionary configuration for SCOPE.

    expected_priors : dict
        A dictionary containing expected parameter names as keys.
    """

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

    if "models" not in configuration.keys():
        raise KeyError("no models found in configuration")

    model_configuration = configuration["models"]

    # Check the aperture names
    aperture_names = get_aperture_names(configuration)
    for aperture in aperture_names:
        if '.' in aperture:
            raise ValueError("aperture name '{0}' cannot contain a full-stop character".format(aperture))

    required_keys = ("dispersion_filenames", "flux_filenames")
    for key in required_keys:
        if key not in model_configuration.keys():
            raise KeyError("no '{0}' found in configuration for models".format(key))

    # Check keys exist in both
    missing_keys = [x for x in model_configuration["dispersion_filenames"].keys() if x not in model_configuration["flux_filenames"]]

    if len(missing_keys) > 0:
        raise KeyError("missing flux filenames for {0} aperture(s)".format(
            ",".join(missing_keys)))

    # Check dispersion filenames exist
    dispersion_maps = {}
    for aperture, dispersion_filename in model_configuration["dispersion_filenames"].iteritems():
        if not os.path.exists(dispersion_filename):
            raise IOError("dispersion filename for {0} aperture does not exist: {1}"
                .format(aperture, dispersion_filename))

        # Load a dispersion map
        dispersion = models.load_model_data(dispersion_filename)
        dispersion_maps[key] = [np.min(dispersion), np.max(dispersion)]

    # Ensure dispersion maps of standards don't overlap
    overlap_wavelength = utils.find_spectral_overlap(dispersion_maps.values())
    if overlap_wavelength is not None:
        raise ValueError("dispersion maps overlap near {0}".format(overlap_wavelength))

    # Try actually loading a model
    model = models.Model(configuration)

    # Check the length of the dispersion map and some random point
    for aperture, dispersion_map in model.dispersion.iteritems():
        n_dispersion_points = len(dispersion_map)

        random_filename = choice(model.flux_filenames[aperture])
        random_flux = models.load_model_data(random_filename)
        n_flux_points = len(random_flux)

        if n_dispersion_points != n_flux_points:
            raise ValueError("number of dispersion points ({0}) and flux points ({1})"
                " does not match for {2} aperture in randomly selected filename: {3}".format(
                n_dispersion_points, n_flux_points, aperture, random_filename))

    return True


def verify_masks(configuration):
    """Verifies any (optional) masks specified in the input configuration."""

    if "masks" not in configuration.keys():
        # Masks are optional
        return True

    # Check the aperture names
    aperture_names = get_aperture_names(configuration)

    for mask_aperture_name in configuration["masks"]:
        if mask_aperture_name not in aperture_names:
            raise ValueError("unrecognised aperture name '{0}' specified in masks".format(mask_aperture_name))

        # We know nothing about the wavelength coverage, so all we can do is check
        # for float/ints
        if not isinstance(configuration["masks"][mask_aperture_name], (tuple, list)):
            raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")

        for region in configuration["masks"][mask_aperture_name]:
            if not isinstance(region, (list, tuple)) or len(region) != 2:
                raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")

            try:
                map(float, region)
            except TypeError:
                raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")

    return True


def verify_weights(configuration):
    """Verifies any (optional) pixel weighting functions specified in the input configuration."""

    if "weights" not in configuration.keys():
        # Weighting functions are optional
        return True

    # Check the aperture names
    aperture_names = get_aperture_names(configuration)

    for aperture in configuration["weights"]:
        if aperture not in aperture_names:
            raise ValueError("unrecognised aperture name '{0}' specified in weights".format(
                aperture))

        # Check is callable
        test_weighting_func = lambda disp, flux: eval(configuration["weights"][aperture],
            {"disp": disp, "flux": flux, "np": np})

        try:
            test_weighting_func(1, 1)
            test_weighting_func(np.arange(10), np.ones(10))

        except:
            raise ValueError("weighting function for {0} aperture is improper".format(aperture))

    return True
        

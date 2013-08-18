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

    verify_models(configuration)

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
    for beam, dispersion_map in toy_model.dispersion.iteritems():
        n_dispersion_points = len(dispersion_map)

        random_filename = choice(toy_model.flux_filenames[beam])
        random_flux = models.load_model_data(random_filename)
        n_flux_points = len(random_flux)

        if n_dispersion_points != n_flux_points:
            raise ValueError("number of dispersion points ({n_dispersion_points}) and flux points ({n_flux_points})"
                " does not match for {beam} beam in randomly selected filename: {random_filename}"
                .format(n_dispersion_points=n_dispersion_points, beam=beam, n_flux_points=n_flux_points,
                    random_filename=random_filename))


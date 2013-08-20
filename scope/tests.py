# coding: utf-8

""" SCOPE tests. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os
import re

from glob import glob

# Third-party
import numpy as np
import pyfits
import scipy
import yaml

# Module imports
import config

folder_path = os.path.abspath(os.path.dirname(__file__))
environment = ['remote', 'localhost'][folder_path.startswith('/Users/andycasey/')]

def test_assert():
    assert True


def test_default_configuration():
    """Verifies that the default configuration file is valid."""
    
    configuration = config.load(os.path.join(folder_path, '../config.yml'))

    # Only verify the models if we are on localhost
    if environment == 'localhost':
        config.verify(configuration)

    else:

        logging.warn("Cannot verify models or priors because no model data exists. Checking normalisation,"
            " smoothing, and doppler configuration only.")

        # Check the normalisation
        normalisation_priors = config.verify_normalisation(configuration)

        # Check the smoothing
        smoothing_priors = config.verify_smoothing(configuration)

        # Check the doppler corrections
        doppler_priors = config.verify_doppler(configuration)


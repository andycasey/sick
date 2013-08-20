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
        
    configuration = config.load(os.path.join(folder_path, '../config.yml'))

    # Verify everything except models
    if environment == 'localhost':
        config.verify_models(configuration)

    # Check the normalisation
    normalisation_priors = config.verify_normalisation(configuration)

    # Check the smoothing
    smoothing_priors = config.verify_smoothing(configuration)

    # Check the doppler corrections
    doppler_priors = config.verify_doppler(configuration)

    # Establish all of the priors    
    priors_to_expect = doppler_priors + smoothing_priors + normalisation_priors

    # Verify that we have priors established for all the priors
    # we expect, and the stellar parameters we plan to solve for
    config.verify_priors(configuration, priors_to_expect)

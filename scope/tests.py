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

from numpy.random import normal

# Module imports
import config

folder_path = os.path.abspath(os.path.dirname(__file__))
environment = ['remote', 'localhost'][folder_path.startswith('/Users/andycasey/')]

def test_assert():
    assert True


def test_sun():

    # Only run this test on localhost
    if environment != 'localhost': return

    import analyze
    from specutils import Spectrum1D

    expected_results = {
        # Parameter: (Expected, Acceptable error)
        'teff': (5777, 150),
        'logg': (4.44, 0.3),
        'feh':  (0.0, 0.2)
    }

    spectrum_filename = os.path.join(folder_path, '../data/sun.ms.fits')
    configuration_filename = os.path.join(folder_path, '../data/sun.yml')

    sun = Spectrum1D.load(spectrum_filename)
    
    # Split into "blue" and "red"
    idx = np.searchsorted(sun.disp, [5700, 7800])
    sun_blue = Spectrum1D(disp=sun.disp[:idx[0]], flux=sun.flux[:idx[0]])
    sun_red = Spectrum1D(disp=sun.disp[idx[1]:], flux=sun.flux[idx[1]:])

    # Assign some uncertainty assuming a uniform S/N ~ 700
    snr = 700
    sun_blue.uncertainty = [1/snr] * len(sun_blue.disp)
    sun_red.uncertainty = [1/snr] * len(sun_red.disp)

    # Analyse the spectrum
    results = analyze.analyze([sun_blue, sun_red], configuration_filename)

    # Check the results are good.
    for parameter, (value, error) in expected_results.iteritems():
        assert np.less_equal(abs(results[parameter] - value), error)

    return results


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


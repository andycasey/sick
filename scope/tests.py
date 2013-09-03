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

logging.basicConfig(level=logging.DEBUG)


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
    results = analyze.analyze([sun_blue, sun_red], configuration_filename, callback=lambda *x: print(("callback:", x)))

    # Check the results are good.
    for parameter, (value, error) in expected_results.iteritems():
        assert np.less_equal(abs(results[parameter] - value), error)

    return results


def test_aaomega():

    if environment != 'localhost': return

    import analyze
    from specutils import Spectrum1D
    import matplotlib.pyplot as plt

    spectra = map(Spectrum1D.load, ['../data/NGC288_blue_8.fits', '../data/NGC288_red_8.fits'])
    configuration_filename = os.path.join(folder_path, '../config.cached.yml')

    def setup():
        # Create figure
        fig = plt.figure()

        # Create axes
        for i in xrange(len(spectra)):
            ax = fig.add_subplot(len(spectra), 1, i + 1)
            # One for model, and one for observed
            ax.plot([], [])
            ax.plot([], [])

    # Setup axes
    setup()

    def callback(total_chi_sq, num_dof, parameters, observed_spectra, model_spectra):

        fig = plt.gcf()
        
        for ax, observed_spectrum, model_spectrum in zip(fig.axes, observed_spectra, model_spectra):

            xlims = [observed_spectrum.disp[0], observed_spectrum.disp[-1]]
            ylims = [0, np.max(observed_spectrum.flux)]

            ax.lines[0].set_data(np.array([observed_spectrum.disp, observed_spectrum.flux]))
            ax.lines[1].set_data(np.array([model_spectrum.disp, model_spectrum.flux]))

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

        plt.draw()
        plt.pause(0.01)
        

    results = analyze.analyze(spectra, configuration_filename, callback=callback)

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


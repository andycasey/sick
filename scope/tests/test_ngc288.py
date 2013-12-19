# coding: utf-8

""" SCOPE tests to use for API development and testing. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os
import sys
from time import time

# Third party
from emcee.utils import MPIPool

# Module imports
import scope


__all__ = ["test_ngc288", "test_all_globular_clusters"]


def test_ngc288():

    # Load the spectra and configuration
    configuration = scope.config.load(os.path.join(os.path.dirname(__file__), "ngc288.cached.yml"))
    blue_spectra = scope.specutils.load_aaomega_multispec(os.path.join(os.path.dirname(__file__), "data/NGC288_blue.fits"))
    red_spectra = scope.specutils.load_aaomega_multispec(os.path.join(os.path.dirname(__file__), "data/NGC288_red.fits"))

    # Initialize the MPI-based pool used for parallelization
    if configuration.get("mpi", False):
        pool = MPIPool()
        if not pool.is_master():
            # Wait for instructions from the master process.
            pool.wait()
            sys.exit(0)
    else:
        pool = None

    # Just get the first spectrum
    spectra = []
    for blue_spectrum, red_spectrum in zip(blue_spectra, red_spectra):
        if blue_spectrum is not None and red_spectrum is not None:
            spectra.append([blue_spectrum, red_spectrum])

    # Analyse the first star
    t_init = time()
    chi_sq, num_dof, posteriors, observed_spectra, model_spectra, masks = scope.analyze_star(spectra,
        configuration, callback=callback, pool=pool)
    logging.info("Analysis took {time_taken:.2f} seconds".format(time_taken=time() - t_init))

    return posteriors


if __name__ == '__main__':
    ngc288 = test_ngc288()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Global hacky approximator stuff. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging

logger = logging.getLogger("sick")

# Now you're thinking with portals!
def init():
    logger.debug("Initialising approximator.")
    global wavelengths, intensities, variances, binning_matrices
    wavelengths = []
    intensities = []
    variances = [0] # In case no variances are set by whatever the model is.
    binning_matrices = [None]
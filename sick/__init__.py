#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" sick, the spectroscopic inference crank """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.2.0"

__all__ = ("models", "plot", "specutils", "utils")

import os
import logging
from numpy import RankWarning
from warnings import simplefilter

# Here I set the default logging level to WARN because -- unless explicitly told
# otherwise -- we don't want the logger to display everything when the API is 
# being used. When the command-line interface is used we will overwrite this
# configuration and set the level to INFO or DEBUG, depending on specified
# verbosity.

# For the moment we will DEBUG ALL THE THINGS.
logging.basicConfig(level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s")

# TODO: Set as warn.
#logging.basicConfig(level=logging.WARN, 
#    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sick")

# Suppress "polyfit may be poorly conditioned" messages
simplefilter("ignore", RankWarning)

# It pains me to have to do this.
if not os.environ.get("DISPLAY", False):
    logger.info("Disabling DISPLAY and forcing Matplotlib to use 'Agg' backend")
    from matplotlib import use
    use("Agg")
    from matplotlib.pyplot import ioff
    ioff()

import models, plot, specutils

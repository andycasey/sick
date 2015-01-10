# coding: utf-8

""" sick, the spectroscopic inference crank """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.10.53"

__all__ = ["Model", "models", "plot", "specutils", "utils"]

import os
import logging
import warnings
from numpy import RankWarning

# It pains me to have to do this.
if not os.environ.get("DISPLAY", False):
    import matplotlib
    matplotlib.use("Agg")

# sick modules, bro.
import models
import plot
import specutils
import utils
from models import Model

# Here I set the default logging level to WARN because -- unless explicitly told
# otherwise -- we don't want the logger to display everything when the API is 
# being used. When the command-line interface is used we will overwrite this
# configuration and set the level to INFO or DEBUG, depending on specified
# verbosity.
logging.basicConfig(level=logging.WARN, 
    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sick")

# Suppress "polyfit may be poorly conditioned" messages
warnings.simplefilter("ignore", RankWarning)
# coding: utf-8

""" Spectroscopic inference of astrophysical quantities """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.092dev"

__all__ = ["config", "Model", "specutils", "Spectrum", "utils"]

import logging

from analysis import solve
from models import Model
from specutils import Spectrum

import specutils

# Here we set the default logging level to WARN because -- unless explicitly told otherwise -- we
# don't want the logger to display everything when the API is being used. When the command-line
# interface is used we will over-ride this configuration and set the level to INFO or DEBUG,
# depending on specified verbosity
logging.basicConfig(level=logging.WARN, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sick")

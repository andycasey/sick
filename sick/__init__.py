# coding: utf-8

""" Spectroscopic inference of astrophysical quantities """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.091dev"

__all__ = ["config", "Model", "specutils", "Spectrum", "utils"]

import logging

from analysis import solve
from models import Model
from specutils import Spectrum

import specutils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("sick")

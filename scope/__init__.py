# coding: utf-8

""" Spectral Comparison and Parameter Evaluation """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.06-alpha"

__all__ = ["config", "Model", "specutils", "utils"]

import logging
from analysis import solve
from models import Model
from specutils import Spectrum

logger = logging.basicConfig(level=logging.INFO)

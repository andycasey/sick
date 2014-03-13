# coding: utf-8

""" Spectral Comparison and Parameter Evaluation """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.01"

__all__ = ["config", "Model", "specutils", "utils"]

import logging
from analysis import solve
from models import Model

logger = logging.basicConfig(level=logging.INFO)
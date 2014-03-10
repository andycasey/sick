# coding: utf-8

""" Spectral Comparison and Parameter Evaluation """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"
__version__ = "0.01"

__all__ = ["config", "models", "specutils", "utils"]

import logging
from analyze import solve

logger = logging.basicConfig(level=logging.INFO)
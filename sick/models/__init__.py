#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

from .interpolation import InterpolationModel
from .cannon import CannonModel
from .create import create
from .base import BaseModel

def Model(filename, **kwargs):

    model = BaseModel(filename, **kwargs)

    is_CannonModel = "cannon_coefficients" in model._configuration["model_grid"]
    klass = CannonModel if is_CannonModel else InterpolationModel

    return klass(filename, **kwargs)
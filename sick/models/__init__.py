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

    _ = model._configuration["model_grid"]
    is_CannonModel = "cannon_data" in _ \
        or _.get("model_type", "").lower() == "cannon"
    
    klass = CannonModel if is_CannonModel else InterpolationModel
    
    return klass(filename, **kwargs)
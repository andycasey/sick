#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Interpolation Model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging

import numpy as np
from scipy import interpolate

import generate
from model import Model
from .. import specutils

logger = logging.getLogger("sick")

class InterpolationModel(Model):

    def _initialise_approximator(self, closest_point=None,
        wavelengths_required=None, rescale=True, **kwargs):
        """
        Initialise a spectrum interpolator.
        """

        if self._initialised and not kwargs.get("force", False):
            logger.debug("Ignoring call to re-initialise approximator because "
                "we already have.")
            return self._subset_bounds


        logger.info("Initialising approximator near {0}".format(closest_point))

        N = self.grid_points.size
        dtype = [(name, '<f8') for name in self.grid_points.dtype.names]
        grid_points = self.grid_points.astype(dtype).view(float).reshape(N, -1)

        # If closest_point is given then we will slice a small part of the grid.
        if closest_point is not None:

            default = 1.0
            grid_subset = self._configuration.get("settings", 
                { "grid_subset": default }).get("grid_subset", default)

            # Use nearest N points in interpolator.
            # If grid subset is a fraction, scale it to real numbers.
            if 1 >= grid_subset > 0:
                grid_subset = int(np.round(grid_subset * N))

            logger.debug("Using {0} nearest points for interpolator".format(
                grid_subset))

            # Use closest N points.
            distance = np.sum(np.abs(grid_points - closest_point)/
                np.ptp(grid_points, axis=0), axis=1)
            grid_indices = np.argsort(distance)[:grid_subset]

        else:
            grid_indices = np.ones(N, dtype=bool)

        # What wavelengths will be required? Assume all if not specified.
        if wavelengths_required is not None:
            mask = np.zeros(self.wavelengths.size, dtype=bool)
            for start, end in wavelengths_required:
                idx = np.clip(
                    self.wavelengths.searchsorted([start, end]) + [0, 1],
                    0, self.wavelengths.size)
                mask[idx[0]:idx[1]] = True
        else:
            mask = np.ones(self.wavelengths.size, dtype=bool)

        # Apply model masks.
        mask *= self._model_mask()

        # Slice small part of the intensities grid.
        intensities = np.memmap(
            self._configuration["model_grid"]["intensities"],
            dtype="float32", mode="r", shape=(N, self.wavelengths.size))
        subset = np.copy(intensities[grid_indices, :])
        subset[:, ~mask] = np.nan
        del intensities

        # Create an interpolator.
        try:
            interpolator = interpolate.LinearNDInterpolator(
                grid_points[grid_indices], subset, rescale=rescale)

        except TypeError:
            logger.warn("Could not rescale the LinearNDInterpolator because "\
                "you need a newer version of scipy")
            interpolator = interpolate.LinearNDInterpolator(
                grid_points[grid_indices], subset)

        generate.init()
        generate.wavelengths.append(self.wavelengths)
        generate.intensities.append(interpolator)
        
        self._initialised = True

        # Return the subset boundaries of the grid.
        self._subset_bounds = {}
        for name in self.grid_points.dtype.names:
            points = self.grid_points[name][grid_indices]
            self._subset_bounds[name] = (min(points), max(points))

        return self._subset_bounds


    def _approximate_intensities(self, theta, data, debug=False, **kwargs):
        """
        Intepolate model intensities at the given data points.
        """

        if kwargs.get("__intensities", None) is not None:
            # We have been passed some intensities to use. This typically
            # happens for the initial_theta estimate.
            model_wavelengths = self.wavelengths
            model_intensities = kwargs.pop("__intensities")
            model_variances = np.zeros_like(model_wavelengths)

        else:
            # Generate intensities at the astrophysical point.   
            try:
                # Get the wavelengths.
                model_wavelengths = generate.wavelengths[-1]
                
                # Generate intensities.
                func = generate.intensities[-1]
                model_intensities = func(*[theta.get(p, np.nan) \
                    for p in self.grid_points.dtype.names]).flatten()
                model_variances = np.zeros_like(model_wavelengths)

            except:
                if debug: raise
                # Return dummy flux arrays with nan's.
                return [np.nan * np.ones(s.disp.size) for s in data]

        return (model_wavelengths, model_intensities, model_variances)


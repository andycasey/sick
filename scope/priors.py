# coding: utf-8

""" Priors """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["implicit", "explicit"]

# Standard library
import logging
import os
from time import time

# Third-party
import numpy as np
import numpy.random as random

# Module
import specutils

logger = logging.getLogger(__name__.split(".")[0])

def implicit(model, observations, size=1):
    """ Generate implicit priors for all dimensions in the given model """

    priors = []
    while size > len(priors):

        walker_prior = []
        normalisation_coefficients = {}

        failed_v_rad_indices = []
        success_v_rad_indices = []
        for i, dimension in enumerate(model.dimensions):

            # Is this a grid dimension?
            # If it is, let's assume it's uniformly distributed
            if dimension in model.grid_points.dtype.names:
                walker_prior.append(random.uniform(*model.grid_boundaries[dimension]))
                continue

            # The grid dimensions come first in model.dimensions, so once the length
            # of walker_prior matches model.grid_points.dtype.names, we have what we
            # need to interpolate a flux. The interpolated flux will can be used for
            # the other priors
            if len(model.grid_points.dtype.names) == len(walker_prior):
                # Attempt to interpolate a flux
                interpolated_fluxes = model.interpolate_flux(walker_prior)

                non_finite = lambda _: np.all(~np.isfinite(_))
                if all(map(non_finite, interpolated_fluxes.values())): 
                    break

            # Velocities
            if dimension.startswith("doppler_shift."):
                # Assumes a velocity, not a redshift
                #velocity = random.normal(0, 100)
                #walker_prior.append(velocity)

                aperture = dimension.split(".")[1]
                observed_aperture = observations[model.apertures.index(aperture)]
                model_aperture = specutils.Spectrum1D(disp=model.dispersion[aperture], flux=interpolated_fluxes[aperture])
                try:
                    v_rad, u_v_rad, r = observed_aperture.cross_correlate(model_aperture)

                except (ValueError, ):
                    failed_v_rad_indices.append(i)
                    walker_prior.append(random.normal(0, 100))
                    
                else:
                    if abs(v_rad) > 500:
                        failed_v_rad_indices.append(i)
                        walker_prior.append(random.normal(0, 100))

                    else:
                        success_v_rad_indices.append(i)
                        walker_prior.append(v_rad)

            # Smoothing
            elif dimension.startswith("smooth_model_flux."):

                aperture = dimension.split(".")[1]

                walker_prior.append({"blue": 1.8, "red": 0.8}[aperture])

                """
                # Estimate this by the observed spectral resolution
                observed_aperture = observations[model.apertures.index(aperture)]
                observed_spectral_resolution = observed_aperture.disp[1:]/np.diff(observed_aperture.disp)

                model_spectral_resolution = model.dispersion[aperture][1:]/np.diff(model.dispersion[aperture])

                resolution_difference = np.mean(model_spectral_resolution)/np.mean(observed_spectral_resolution)

                resolution_difference /= (2 * (2 * np.log(2))**0.5)

                walker_prior.append(random.normal(resolution_difference, 0.1 * resolution_difference))
                """

            # Normalisation
            elif dimension.startswith("normalise_observed."):

                # We will use the interpolated flux to determine approximate normalisation
                # coefficients
                aperture = dimension.split(".")[1]
                
                if model.configuration["normalise_observed"][aperture].get("method", "polynomial") == "polynomial":
                
                    if aperture not in normalisation_coefficients.keys():
                        model_flux = interpolated_fluxes[aperture]

                        order = model.configuration["normalise_observed"][aperture]["order"]
                        aperture_index = model.apertures.index(aperture)
                        observed_aperture = observations[aperture_index]

                        # Get masks
                        # TODO

                        interpolated_model_flux = np.interp(observed_aperture.disp,
                            model.dispersion[aperture], model_flux, left=np.nan, right=np.nan)
                        continuum = observed_aperture.flux/interpolated_model_flux

                        finite = np.isfinite(continuum)
                        coefficients = np.polyfit(observed_aperture.disp[finite], continuum[finite], order)
                        
                        normalisation_coefficients[aperture] = coefficients

                    coefficient = int(dimension.split(".")[2].lstrip("a"))
                    walker_prior.append(normalisation_coefficients[aperture][coefficient])

                else: #spline
                    # Smoothing value
                    m = sum(np.isfinite(interpolated_fluxes[aperture]))
                    walker_prior.append(m)

            # Jitter
            elif dimension == "jitter" or dimension.startswith("jitter."):
                # Uniform between -10 and 1
                walker_prior.append(random.uniform(-10, 1))

            else:
                raise RuntimeError("don't know how to generate implicit priors from dimension {0}".format(dimension))

        else:
            for index in failed_v_rad_indices:
                if len(success_v_rad_indices) > 0:
                    walker_prior[index] = walker_prior[success_v_rad_indices[0]]

            priors.append(walker_prior)

    priors = np.array(priors)
    if size == 1:
        priors = priors.flatten()

    return priors


def explicit(model, observations):
    """ Initialise the priors (or initial conditions) for the analysis """

    walker_priors = []
    initial_normalisation_coefficients = {}

    walkers = model.configuration["solver"].get("walkers", 1)

    if model.configuration.has_key("priors"):
        while walkers > len(walker_priors):

            current_walker = []
            interpolated_flux = {}
            initial_normalisation_coefficients = {}
            for j, dimension in enumerate(model.dimensions):

                # Have we just finished doing the model dimensions?
                # If so then we can interpolate to a model flux
                if len(current_walker) == len(model.grid_points.dtype.names):
                    interpolated_flux = model.interpolate_flux(current_walker)

                    if np.all(~np.isfinite(interpolated_flux.values()[0])):
                        # None of the flux in the first beam are finite.

                        break

                # Jitter
                if dimension == "jitter" or dimension.startswith("jitter."):
                    # Uniform prior between 0 and 1
                    current_walker.append(random.uniform(-10, 1))
                    continue

                # Implicit priors
                if dimension.startswith("normalise_observed."):

                    aperture = dimension.split(".")[1]
                    coefficient_index = int(dimension.split(".")[-1].lstrip("a"))

                    # If we're at this stage we should have grid point dimensions
                    if aperture not in initial_normalisation_coefficients:

                        index = model.apertures.index(aperture)
                        order = model.configuration["normalise_observed"][aperture]["order"]

                        spectrum = observations[index]
                        
                        # Get the full range of spectra that will be normalised
                        if "masks" in model.configuration and aperture in model.configuration["masks"]:
                            
                            ranges = np.array(model.configuration["masks"][aperture])
                            min_range, max_range = np.min(ranges), np.max(ranges)
                            range_indices = np.searchsorted(spectrum.disp, [min_range, max_range])

                            flux_indices = np.zeros(len(spectrum.disp), dtype=bool)
                            flux_indices[range_indices[0]:range_indices[1]] = True
                            flux_indices *= np.isfinite(spectrum.flux)
                            
                            logger.debug("Normalising from {1:.0f} to {2:.0f} Angstroms in {0} aperture".format(
                                aperture, np.min(spectrum.disp[flux_indices]), np.max(spectrum.disp[flux_indices])))
                        else:
                            flux_indices = np.isfinite(spectrum.flux)

                        # Fit the spectrum with a polynomial of order X
                        resampled_interpolated_flux = np.interp(spectrum.disp[flux_indices], model.dispersion[aperture],
                            interpolated_flux[aperture])
                        coefficients = np.polyfit(spectrum.disp[flux_indices], spectrum.flux[flux_indices]/resampled_interpolated_flux, order)
                        
                        # Save the coefficients and variances
                        initial_normalisation_coefficients[aperture] = coefficients
                    
                    coefficient = initial_normalisation_coefficients[aperture][coefficient_index]
                    current_walker.append(coefficient)

                    continue

                # Explicit priors
                prior_value = model.configuration["priors"][dimension]
                try:
                    prior_value = float(prior_value)

                except:

                    # We probably need to evaluate this.
                    if prior_value.lower() == "uniform":
                        # Only works on stellar parameter values.
                        possible_points = model.grid_points[dimension].view(np.float)

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with uniform distribution between {1:.2e} and {2:.2e}".format(
                                dimension, np.min(possible_points), np.max(possible_points)))
                        current_walker.append(random.uniform(np.min(possible_points), np.max(possible_points)))

                    elif prior_value.lower().startswith("normal"):
                        mu, sigma = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with a normal distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
                                dimension, mu, sigma))
                        current_walker.append(random.normal(mu, sigma))

                    elif prior_value.lower().startswith("uniform"):
                        minimum, maximum = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with a uniform distribution between {1:.2e} and {2:.2e}".format(
                                dimension, minimum, maximum))
                        current_walker.append(random.uniform(minimum, maximum))

                    elif prior_value.lower() == "cross_correlate":
                        current_walker.append(random.normal(0, 100))

                        #aperture = dimension.split(".")[1]
                        #observed_spectrum = observations[model.apertures.index(aperture)]

                        #model_aperture_spectrum = specutils.Spectrum1D(model.dispersion[aperture], interpolated_flux[aperture])
                        #v_rad, u_v_rad, R = observed_spectrum.cross_correlate(model_aperture_spectrum)
                        #current_walker.append(random.normal(v_rad, u_v_rad))

                    else:
                        raise TypeError("prior type not valid for {dimension}".format(dimension=dimension))

                else:
                    if len(walker_priors) == 0: # Only print initialisation for the first walker
                        logger_fn = logger.info if walkers == 1 else logger.warn
                        logger_fn("Initialised {0} parameter as a single value: {1:.2e}".format(dimension, prior_value))

                    current_walker.append(prior_value)

            
            # Was the walker actually valid?
            if len(current_walker) > 0:

                # Add the walker
                if walkers == 1:
                    walker_priors = current_walker
                
                else:
                    walker_priors.append(current_walker)

        walker_priors = np.array(walker_priors)

    else:
        walker_priors = implicit_prior(model, observations, size=walkers)

    return walker_priors
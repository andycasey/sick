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
<<<<<<< HEAD
from scipy import ndimage, optimize, interpolate
=======
from scipy import ndimage, optimize, interpolate, stats
>>>>>>> 9c89b5a3373b0ccf030722c2fdece2a59476289c

# Module
import specutils

logger = logging.getLogger(__name__.split(".")[0])


def __cross_correlate__(args):
    """
    Return a redshift by cross correlation of a model spectra and observed spectra.
    """

    dispersion, observed_flux, model_flux = args

    # Be forgiving, although we shouldn't have to be.
    N = np.min(map(len, [dispersion, observed_flux, model_flux]))

    # Ensure an even number of points
    if N % 2 > 0:
        N -= 1

    dispersion = dispersion[:N]
    observed_flux = observed_flux[:N]
    model_flux = model_flux[:N]

    assert len(dispersion) == len(observed_flux)
    assert len(observed_flux) == len(model_flux)
    
    # Set up z array
    m = len(dispersion) / 2
    z_array = dispersion/dispersion[N/2] - 1.0
    
    # Apodize edges
    edge_buffer = 0.1 * (dispersion[-1] - dispersion[0])
    low_w_indices = np.nonzero(dispersion < dispersion[0] + edge_buffer)[0]
    high_w_indices = np.nonzero(dispersion > dispersion[-1] - edge_buffer)[0]

    apod_curve = np.ones(N, dtype='d')
    apod_curve[low_w_indices] = (1.0 + np.cos(np.pi*(1.0 - (dispersion[low_w_indices] - dispersion[0])/edge_buffer)))/2.
    apod_curve[high_w_indices] = (1.0 + np.cos(np.pi*(1.0 - (dispersion[-1] - dispersion[high_w_indices])/edge_buffer)))/2.

    apod_observed_flux = observed_flux * apod_curve
    apod_model_flux = model_flux * apod_curve

    fft_observed_flux = np.fft.fft(apod_observed_flux)
    fft_model_flux = np.fft.fft(apod_model_flux)
    model_flux_corr = (fft_observed_flux * fft_model_flux.conjugate())/np.sqrt(np.inner(apod_observed_flux, apod_observed_flux) * np.inner(apod_model_flux, apod_model_flux))

    correlation = np.fft.ifft(model_flux_corr).real

    # Reflect about zero
    ccf = np.zeros(N)
    ccf[:N/2] = correlation[N/2:]
    ccf[N/2:] = correlation[:N/2]
    
    # Get height and redshift of best peak
    h = ccf.max()
    
    # Scale the CCF
    ccf -= ccf.min()
    ccf *= (h/ccf.max())

    z_best = z_array[ccf.argmax()]    
    z_err = (np.ptp(z_array[np.where(ccf >= 0.5*h)])/2.35482)**2

    return np.random.normal(z_best, z_err)


def prior(model, observations, size=1):

    # Set the default priors:
    prior_distributions = dict(zip(
        model.grid_points.dtype.names,
        ["uniform({0}, {1})".format(*model.grid_boundaries[d]) \
            for d in model.grid_points.dtype.names]))
    
    # Default doppler shift priors
    prior_distributions.update(dict(
        [("z.{}".format(c), "cross_correlate({})".format(c)) for c in model.channels]
    ))

    # Default jitter priors
    prior_distributions.update(dict(
        [("jitter.{}".format(c), "uniform(-10, 1)") for c in model.channels]
    ))

    # Get explicit priors
    prior_distributions.update(model.configuration.get("priors", {}))

    # Environment variables for explicit priors
    # The channel names will be passed to contain all the information required
    # for cross-correlation

    env = { "locals": None, "globals": None, "__name__": None, "__file__": None,
        "__builtins__": None, "normal": random.normal, "uniform": random.uniform,
        "cross_correlate": __cross_correlate__ }

    priors = []
    while size > len(priors):

        current_prior = []
        model_intensities = {}
        continuum_coefficients = {}

        evaluated_priors = {}
        scaled_observations = {}

        for i, dimension in enumerate(model.dimensions):

            # Have we got priors for all grid points?
            if len(current_prior) == len(model.grid_points.dtype.names):

                # Interpolate intensities at this point
                model_intensities.update(model.interpolate_flux(current_prior))

                # Check the intensities are finite, otherwise move on
                if not np.all(map(np.all, map(np.isfinite, model_intensities.values()))):
                    break

                # Smooth the model intensities if required
                for channel in model_intensities.keys():
                    if "convolve.{}".format(channel) in model.dimensions:

                        # We have to evaluate this prior now
                        sigma = eval(prior_distributions["convolve.{}".format(channel)], env)
                        kernel = (sigma/(2 * (2*np.log(2))**0.5))/np.mean(np.diff(model.dispersion[channel]))
                        #model_intensities[channel] = ndimage.gaussian_filter1d(model_intensities[channel], kernel)

                        evaluated_priors["convolve.{}".format(channel)] = kernel

                # Get continuum knots/coefficients for each aperture
                for channel in model_intensities.keys():
                    observed_channel = observations[model.channels.index(channel)]

                    continuum = observed_channel.flux/np.interp(observed_channel.disp,
                        model.dispersion[channel], model_intensities[channel], left=np.nan,
                        right=np.nan)

                    finite = np.isfinite(continuum)
                    method = model.configuration["normalise"][channel]["method"]

                    # Re-interpolate the observed fluxes where they are nans
                    finite_observed_fluxes = np.isfinite(observed_channel.flux)
                    cleaned_observed_flux = np.interp(observed_channel.disp,
                        observed_channel.disp[finite_observed_fluxes], observed_channel.flux[finite_observed_fluxes])

                    if method == "polynomial":
                        # Fit polynomial coefficients
                        order = model.configuration["normalise"][channel]["order"]
                        continuum_coefficients[channel] = np.polyfit(observed_channel.disp[finite], continuum[finite], order)

                        # Re-bin onto log-lambda scale
                        log_delta = np.diff(observed_channel.disp).min()
                        wl_min, wl_max = observed_channel.disp.min(), observed_channel.disp.max()
                        log_observed_dispersion = np.exp(np.arange(np.log(wl_min), np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

                        # Scale the intensities to the data
                        interpolated_model_intensities = np.interp(log_observed_dispersion, model.dispersion[channel],
                            model_intensities[channel], left=np.nan, right=np.nan)

                        observed_scaled_intensities = cleaned_observed_flux \
                            / np.polyval(continuum_coefficients[channel], observed_channel.disp)

                        interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                            observed_scaled_intensities, left=1, right=1)

                        # Get only finite overlap
                        finite = np.isfinite(interpolated_model_intensities)

                        env.update({channel:
                            (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])
                        })

                    elif method == "spline":
                        num_knots = model.configuration["normalise"][channel]["knots"]

                        # Determine knot spacing
                        common_finite_dispersion = observed_channel.disp[finite]
                        knot_spacing = np.ptp(common_finite_dispersion)/(num_knots + 1)

                        # Determine the knots
                        continuum_coefficients[channel] = np.arange(common_finite_dispersion[0],
                            common_finite_dispersion[-1], knot_spacing)[:num_knots]

            # Have we already evaluated this dimension?
            if dimension in evaluated_priors.keys():
                current_prior.append(evaluated_priors[dimension])
                continue

            # Is there an explicitly specified distribution for this dimension?
            specified_prior = prior_distributions.get(dimension, "").lower()

            # Do we have an explicit prior?
            if len(specified_prior) > 0:

                # Evaluate the prior given the environment information
                current_prior.append(eval(specified_prior, env))
                continue

            # These are all implicit priors from here onwards.
    
            # Smoothing
            if dimension.startswith("convolve."):
                raise NotImplementedError("no estimate of convolving a priori yet")

            # Normalise
            elif dimension.startswith("normalise."):
                # Get the coefficient
                channel = dimension.split(".")[1]
                coefficient_index = int(dimension.split(".")[2][1:])
                coefficient_value = continuum_coefficients[channel][coefficient_index]

                # Polynomial coefficients will introduce their own scatter
                # if the method is a spline, we should produce some scatter around the points
                method = model.configuration["normalise"][channel]["method"]
                if method == "polynomial":
                    current_prior.append(coefficient_value)

                elif method == "spline":
                    # Get the difference between knot points
                    knot_sigma = np.mean(np.diff(continuum_coefficients[channel]))/10.
                    current_prior.append(random.normal(coefficient_value, knot_sigma))

        # Check that we have the full number of walker values
        if len(current_prior) == len(model.dimensions):
            priors.append(current_prior)

    return np.array(priors) if size > 1 else np.array(priors).flatten()


def implicit(model, observations, size=1):
    """
    Generate implicit priors for all dimensions in the given model.
    """

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
            if dimension.startswith("z."):
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
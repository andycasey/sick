# coding: utf-8

""" Priors """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["prior"]

# Standard library
import logging
import os
from itertools import chain

# Third-party
import numpy as np
import numpy.random as random
from scipy import interpolate

logger = logging.getLogger(__name__.split(".")[0])

def __cross_correlate__(args):
    """
    Return a redshift by cross correlation of a model spectra and observed spectra.

    Args:
        dispersion (array-like object): The dispersion points of the observed and template fluxes.
        observed_flux (array-like object): The observed fluxes.
        model_flux (array-like object): The template fluxes.
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

    return z_best


def prior(model, observations):
    """
    Yield a prior theta for the model and data provided.

    Args:
        model (sick.models.Model object): The model class.
        observations (list of specutils.Spectrum1D objects): The observed spectra.
        size (integer): The number of priors to return.
    """

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

    # Default outlier distribution priors
    prior_distributions.update({"Pb": "uniform(0, 1)"})
    all_fluxes = np.array(list(chain(*[each.flux for each in observations])))
    all_fluxes = all_fluxes[np.isfinite(all_fluxes)]

    prior_distributions.update({
        "Yb": "normal({0}, 0.5 * {0})".format(np.median(all_fluxes)),
        "Vb": "normal({0}, 0.5 * {0})".format(np.std(all_fluxes)**2)
        })

    # Get explicit priors
    prior_distributions.update(model.configuration.get("priors", {}))

    # Environment variables for explicit priors
    # The channel names will be passed to contain all the information required
    # for cross-correlation

    env = { "locals": None, "globals": None, "__name__": None, "__file__": None,
        "__builtins__": None, "normal": random.normal, "uniform": random.uniform,
        "cross_correlate": __cross_correlate__ }

    current_prior = []
    model_intensities = {}
    continuum_parameters = {}

    evaluated_priors = {}
    scaled_observations = {}

    for i, dimension in enumerate(model.dimensions):

        # Have we got priors for all grid points?
        if len(current_prior) == len(model.grid_points.dtype.names):

            # Interpolate intensities at this point
            try:
                model_intensities.update(model.interpolate_flux(current_prior))
            except (IndexError, ValueError) as e:
                break

            # Check the intensities are finite, otherwise move on
            if not np.all(map(np.all, map(np.isfinite, model_intensities.values()))):
                break

            # Smooth the model intensities if required
            for channel in model_intensities.keys():
                if "convolve.{}".format(channel) in model.dimensions:

                    # We have to evaluate this prior now
                    sigma = eval(prior_distributions["convolve.{}".format(channel)], env)
                    kernel = (sigma/(2 * (2*np.log(2))**0.5))/np.mean(np.diff(model.dispersion[channel]))
                    
                    evaluated_priors["convolve.{}".format(channel)] = sigma

            # Get continuum knots/coefficients for each aperture
            for channel in model_intensities.keys():
                observed_channel = observations[model.channels.index(channel)]

                continuum = observed_channel.flux/np.interp(observed_channel.disp,
                    model.dispersion[channel], model_intensities[channel], left=np.nan,
                    right=np.nan)

                finite_continuum = np.isfinite(continuum)

                if model.configuration.get("normalise", False) and channel in model.configuration["normalise"]:
                    method = model.configuration["normalise"][channel]["method"]

                else:
                    method = None

                # Re-interpolate the observed fluxes where they are nans
                finite_observed_fluxes = np.isfinite(observed_channel.flux)
                cleaned_observed_flux = np.interp(observed_channel.disp,
                    observed_channel.disp[finite_observed_fluxes], observed_channel.flux[finite_observed_fluxes])

                # Re-bin onto log-lambda scale
                log_delta = np.diff(observed_channel.disp).min()
                wl_min, wl_max = observed_channel.disp.min(), observed_channel.disp.max()
                log_observed_dispersion = np.exp(np.arange(np.log(wl_min), np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

                # Scale the intensities to the data
                interpolated_model_intensities = np.interp(log_observed_dispersion, model.dispersion[channel],
                    model_intensities[channel], left=np.nan, right=np.nan)

                # Get only finite overlap
                finite = np.isfinite(interpolated_model_intensities)

                if method == "polynomial":
                    # Fit polynomial coefficients
                    order = model.configuration["normalise"][channel]["order"]
                    continuum_parameters[channel] = np.polyfit(observed_channel.disp[finite_continuum], continuum[finite_continuum], order)

                    # Transform the observed data
                    observed_scaled_intensities = cleaned_observed_flux \
                        / np.polyval(continuum_parameters[channel], observed_channel.disp)
                    interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                        observed_scaled_intensities, left=1, right=1)

                    env.update({channel:
                        (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])
                    })

                elif method == "spline":
                    num_knots = model.configuration["normalise"][channel]["knots"]

                    # Determine knot spacing
                    finite_continuum = np.isfinite(continuum)
                    knot_spacing = np.ptp(observed_channel.disp[finite_continuum])/(num_knots + 1)
                    continuum_parameters[channel] = np.arange(observed_channel.disp[finite_continuum][0] + knot_spacing,
                        observed_channel.disp[finite_continuum][-1] + knot_spacing, knot_spacing)[:num_knots]

                    tck = interpolate.splrep(observed_channel.disp[finite_continuum], continuum[finite_continuum],
                        w=1./np.sqrt(observed_channel.variance[finite_continuum]), t=continuum_parameters[channel])

                    # Transform the observed data
                    observed_scaled_intensities = cleaned_observed_flux \
                        / interpolate.splev(observed_channel.disp, tck)
                    interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                        observed_scaled_intensities, left=1, right=1)

                    env.update({channel:
                        (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])
                    })

                else:
                    # No normalisation specified, but we might be required to get a cross-correlation prior.
                    interpolated_observed_intensities = np.interp(log_observed_dispersion, observed_channel.disp,
                        cleaned_observed_flux, left=1, right=1)
                    env.update({channel:
                        (log_observed_dispersion[finite], interpolated_observed_intensities[finite], interpolated_model_intensities[finite])})

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
            coefficient_value = continuum_parameters[channel][coefficient_index]

            # Polynomial coefficients will introduce their own scatter
            # if the method is a spline, we should produce some scatter around the points
            method = model.configuration["normalise"][channel]["method"]
            if method == "polynomial":
                current_prior.append(coefficient_value)

            elif method == "spline":
                # Get the difference between knot points
                knot_sigma = np.mean(np.diff(continuum_parameters[channel]))/10.
                current_prior.append(random.normal(coefficient_value, knot_sigma))

    # Check that we have the full number of walker values
    if len(current_prior) == len(model.dimensions):
        return current_prior

    else:
        # To understand recursion, first you must understand recursion.
        return prior(model, observations)
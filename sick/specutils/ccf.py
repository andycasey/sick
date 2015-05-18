# coding: utf-8

""" Cross-correlation functionality for 1D spectra. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["Spectrum1D"]

import numpy as np
from .sample import resample

from astropy.constants import c as speed_of_light

c = speed_of_light.to("km/s").value

def cross_correlate(observed, template_dispersion, template_fluxes,
    rebin="template", wavelength_range=None, continuum_degree=-1,
    apodize=0.10, rescale=True, full_output=False):
    """
    Cross-correlate the observed spectrum against template fluxes.
    """

    template_fluxes = np.atleast_2d(template_fluxes)
    template_dispersion = np.array(template_dispersion)
    if template_dispersion.shape[0] != template_fluxes.shape[1]:
        raise ValueError("template dispersion must have size (N_pixels,) and "\
            "template fluxes must have size (N_models, N_pixels)")

    try:
        continuum_degree = int(continuum_degree)
    except (TypeError, ValueError):
        raise TypeError("continuum order must be an integer-like object")

    if not (1 > apodize >= 0):
        raise ValueError("apodize fraction must be between 0 and 1")

    if rebin.lower() == "template":
        # Put the template fluxes onto the observed dispersion map.
        dispersion = observed.disp
        template_flux = template_fluxes \
            * resample(template_dispersion, observed.disp)
        observed_flux = observed.flux.copy()

    elif rebin.lower() == "observed":
        # Put the observed fluxes onto the template dispersion map.
        dispersion = template_dispersion
        template_flux = template_fluxes
        observed_flux = observed.flux.copy() \
            * resample(observed.disp, template_dispersion)
        
    else:
        raise ValueError("rebin must be either `template` or `observed`")

    if wavelength_range is not None:
        if not isinstance(wavelength_range, (tuple, list, np.ndarray)) \
        or len(wavelength_range) != 2:
            raise TypeError("wavelength range must be a two length tuple")

        indices = np.clip(
            template_dispersion.searchsorted(wavelength_range) + [0, 1],
            0, template_dispersion.size)
        template_dispersion = template_dispersion[indices[0]:indices[1]]
        template_flux = template_flux[indices[0]:indices[1]]
        observed_flux = observed_flux[indices[0]:indices[1]]

    # Ensure an even number of points.
    N = dispersion.size
    N = N - 1 if N % 2 > 0 else N
    dispersion = dispersion[:N]
    observed_flux = observed_flux[:N]
    template_flux = template_flux[:, :N]

    N_templates = template_fluxes.shape[0]

    # Interpolate over non-finite pixels.
    finite = np.isfinite(observed_flux)
    observed_flux[~finite] = np.interp(dispersion[~finite],
        dispersion[finite], observed_flux[finite])

    # Continuum.
    if continuum_degree >= 0:
        coefficients = np.polyfit(dispersion, observed_flux,
            continuum_degree)
        observed_flux /= np.polyval(coefficients, dispersion)

    # Scale the flux level to that the template intensities
    if rescale:
        observed_flux = (observed_flux * template_flux.ptp()) \
            + template_flux.min()

    # Apodize edges.
    edge_buffer = apodize * (dispersion[-1] - dispersion[0])
    low_w_indices = np.nonzero(dispersion < dispersion[0] + edge_buffer)[0]
    high_w_indices = np.nonzero(dispersion > dispersion[-1] - edge_buffer)[0]

    apod_curve = np.ones(N, dtype='d')
    apod_curve[low_w_indices] = (1.0 + np.cos(np.pi*(
        1.0 - (dispersion[low_w_indices] - dispersion[0])/edge_buffer)))/2.
    apod_curve[high_w_indices] = (1.0 + np.cos(np.pi*(
        1.0 - (dispersion[-1] - dispersion[high_w_indices])/edge_buffer)))/2.

    apod_observed_flux = observed_flux * apod_curve
    apod_template_flux = template_flux * apod_curve

    fft_observed_flux = np.fft.fft(apod_observed_flux)
    fft_template_flux = np.fft.fft(apod_template_flux)
    template_flux_corr = (fft_observed_flux * fft_template_flux.conjugate())
    template_flux_corr /= np.sqrt(
        np.inner(apod_observed_flux, apod_observed_flux))

    z_array = np.array(dispersion.copy())/dispersion[N/2] - 1.0

    z = np.ones(N_templates) * np.nan
    z_err = np.ones(N_templates) * np.nan
    R = np.ones(N_templates) * np.nan
    for i in range(N_templates):

        denominator = np.sqrt(
            np.inner(apod_template_flux[i, :], apod_template_flux[i, :]))
        flux_correlation = template_flux_corr[i, :] / denominator 
        correlation = np.fft.ifft(flux_correlation).real

        # Reflect about zero
        ccf = np.zeros(N)
        ccf[:N/2] = correlation[N/2:]
        ccf[N/2:] = correlation[:N/2]

        # Get height and redshift of best peak
        h = ccf.max()

        # Scale the CCF
        ccf -= ccf.min()
        ccf *= (h/ccf.max())
        
        z[i] = z_array[ccf.argmax()]
        z_err[i] = (np.ptp(z_array[np.where(ccf >= 0.5*h)])/2.35482)**2
        R[i] = h
        
    return (z * c, z_err * c, R)



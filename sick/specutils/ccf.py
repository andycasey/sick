# coding: utf-8

""" Cross-correlation functionality for 1D spectra. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["Spectrum1D"]

import numpy as np
from scipy.optimize import curve_fit
from astropy.constants import c as speed_of_light

from .sample import resample

c = speed_of_light.to("km/s").value

def cross_correlate(observed, template_dispersion, template_fluxes,
    rebin="template", wavelength_range=None, continuum_degree=-1,
    apodize=0.10, rescale=False, full_output=False):
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
        observed_ivar = observed.ivariance.copy()

    elif rebin.lower() == "observed":
        # Put the observed fluxes onto the template dispersion map.
        dispersion = template_dispersion
        template_flux = template_fluxes
        mat = resample(observed.disp, template_dispersion)
        observed_flux = observed.flux.copy() * mat
        observed_ivar = observed.ivariance.copy() * mat
        
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

    N_templates = template_fluxes.shape[0]

    # Clip out the non-finite edges, if they exist.
    _ = np.where(np.isfinite(template_flux[0] * observed_flux))[0]
    l_idx, u_idx = _[0], _[-1]

    dispersion = dispersion[l_idx:u_idx]
    observed_flux = observed_flux[l_idx:u_idx]
    observed_ivar = observed_ivar[l_idx:u_idx]
    template_flux = template_flux[:, l_idx:u_idx]


    # Ensure an even number of points.
    N = u_idx - l_idx
    N = N - 1 if N % 2 > 0 else N
    dispersion = dispersion[:N]
    observed_flux = observed_flux[:N]
    observed_ivar = observed_ivar[:N]
    template_flux = template_flux[:, :N]

    # Interpolate over non-finite pixels.
    finite = np.isfinite(observed_flux)
    observed_flux[~finite] = np.interp(dispersion[~finite], dispersion[finite],
        observed_flux[finite])
    finite = np.isfinite(observed_ivar)
    observed_flux[~finite] = 1e-8

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
        #ccf -= ccf.min()
        #ccf *= (h/ccf.max())
        
        R[i] = h
        z[i] = z_array[ccf.argmax()]
        try:
            z_err[i] = (np.ptp(z_array[np.where(ccf >= 0.5*h)])/2.35482)**2
        except ValueError:
            continue

    # Re-measure the velocity at the best peak.
    index = np.nanargmax(R)

    denominator = np.sqrt(
        np.inner(apod_template_flux[index, :], apod_template_flux[index, :]))

    flux_correlation = template_flux_corr[index, :] / denominator
    correlation = np.fft.ifft(flux_correlation).real

    # FReflect about zero.
    ccf = np.zeros(N)
    ccf[:N/2] = correlation[N/2:]
    ccf[N/2:] = correlation[:N/2]

    h = ccf.max()
    ccf -= ccf.min()
    ccf *= h/ccf.max()

    # Fit +/- 5 pixels
    idx = np.argmax(ccf) - 3, np.argmax(ccf) + 3

    coeffs = np.polyfit(z_array[idx[0]:idx[1]], ccf[idx[0]:idx[1]], 2)
    x_i = np.linspace(z_array[idx[0]], z_array[idx[1]], 1000)
    y_i = np.polyval(coeffs, x_i)

    # Update the value
    z[index] = x_i[y_i.argmax()]

    """
    # Fit the profile peak.
    fwhm = np.ptp(z_array[np.where(ccf >= 0.5)])
    p0 = np.array([z[index], fwhm/2.355, 1.0]) # mu, sigma, peak.
    use = (z_array > (p0[0] - 3 * p0[1])) * ((p0[0] + 3*p0[1]) > z_array)
    x, y = z_array[use], ccf[use]

    # Fit profile:
    f = lambda x, mu, sigma, peak: peak * np.exp(-(mu - x)**2/(2*sigma**2)) 
    
    import matplotlib.pyplot as plt
    plt.plot(z_array, ccf, c='k')
    plt.plot(x, y, c='b')
    plt.plot(x, f(x, *p0), c='r')

    p1 = curve_fit(f, x, y, p0=p0)
    plt.plot(x, f(x, *p1[0]), c='m')

    plt.gca().set_ylim(0,1)
    raise a

    p1 = curve_fit(f, x, y, p0=p0)
    """

    return (z * c, z_err * c, R)



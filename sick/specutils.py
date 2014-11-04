# coding: utf-8

""" Spectroscopic-related utilities for sick """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["Spectrum1D"]

# Standard library
import logging
import os

# Third-party
import numpy as np
import pyfits

logger = logging.getLogger(__name__.split(".")[0])


class Spectrum1D(object):
    """
    This is a temporary class holder for a Spectrum1D object until the
    :class:`astropy.specutils.Spectrum1D` module has advanced sufficiently to replace it.
    """
    
    def __init__(self, disp, flux, variance=None, headers=None):
        """
        Initializes a `Spectrum1D` object with the given dispersion, flux, and
        variance arrays.


        :param disp:
            The dispersion points (e.g., wavelengths) of the spectrum.

        :type disp:
            :class:`np.array`

        :param flux:
            The associated flux points of the spectrum.

        :type flux:
            :class:`np.array`

        :param variance: [optional]
            The variance of the flux points. If not provided, the variance in
            the flux is assumed to be Poisson-distributed.

        :type variance:
            :class:`np.array`

        :param headers: [optional]
            The metadata associated with the spectrum.

        :type headers:
            dict
        """

        if len(disp) != len(flux):
            raise ValueError("dispersion and flux must have the same length")

        if len(disp) == 0:
            raise ValueError("dispersion and flux cannot be empty arrays")
        
        self.disp = disp
        self.flux = flux
        self.variance = variance
        if self.variance is None:
            # Assumed to be Poisson
            self.variance = self.flux.copy()
        self.ivariance = 1.0/self.variance
        if headers is not None:
            self.headers = headers
        else:
            self.headers = {}
        return None


    def copy(self):
        """ Creates a copy of the object """
        variance = self.variance.copy() if self.variance is not None else None
        headers = self.headers.copy() if self.headers is not None else None
        return self.__class__(self.disp.copy(), self.flux.copy(),
            variance=variance, headers=headers)
    

    @classmethod
    def load(cls, filename, **kwargs):
        """Load a Spectrum1D from a given filename.
        
        :param filename:
            The path of the filename to load. Can be either simple 1D FITS files
            or an ASCII filename.

        :type filename:
            str
        """
        
        if not os.path.exists(filename):
            raise IOError("filename {0} does not exist" .format(filename))
        
        if filename.endswith('.fits'):
            image = pyfits.open(filename, **kwargs)
            
            header = image[0].header
            
            # Check for a tabular data structure
            if len(image) > 1 and image[0].data is None:

                names = [name.lower() for name in image[1].data.names]
                dispersion_key = 'wave' if 'wave' in names else 'disp'
                
                disp, flux = image[1].data[dispersion_key], image[1].data['flux']

                if 'error' in names or 'variance' in names:
                    variance_key = 'error' if 'error' in names else 'variance'
                    variance = image[1].data[variance_key]

            else:

                # According to http://iraf.net/irafdocs/specwcs.php ....
                #li = a.headers['LTM1_1'] * np.arange(a.headers['NAXIS1']) + a.headers['LTV1']
                #a.headers['CRVAL1'] + a.headers['CD1_1'] * (li - a.headers['CRPIX1'])

                if np.all([key in header.keys() for key in ('CDELT1', 'NAXIS1', 'CRVAL1')]):
                    disp = header['CRVAL1'] + np.arange(header['NAXIS1']) * header['CDELT1']
            
                if "LTV1" in header.keys():
                    disp -= header['LTV1'] * header['CDELT1']

                #disp -= header['LTV1'] if header.has_key('LTV1') else 0
                flux = image[0].data
                variance = None
            
                # Check for logarithmic dispersion
                if "CTYPE1" in header.keys() and header["CTYPE1"] == "AWAV-LOG":
                    disp = np.exp(disp)

            # Add the headers in
            headers = {}
            for row in header.items():
                key, value = row
                
                # Check the value is valid
                try:
                    str(value)

                except TypeError:
                    logger.debug("Skipping header key {0}".format(key))
                    continue

                if len(key) == 0 or len(str(value)) == 0: continue
    
                if key in headers.keys():
                    if not isinstance(headers[key], list):
                        headers[key] = [headers[key]]
                    
                    headers[key].append(value)

                else:
                    headers[key] = value

            for key, value in headers.iteritems():
                if isinstance(value, list):
                    headers[key] = "\n".join(map(str, value))

        else:
            headers = {}
            try:
                disp, flux, variance = np.loadtxt(filename, unpack=True, **kwargs)
            except:
                disp, flux = np.loadtxt(filename, unpack=True, **kwargs)
            
        return cls(disp, flux, variance=variance, headers=headers)


    def save(self, filename, clobber=True, **kwargs):
        """
        Save the `Spectrum1D` object to the specified filename.
        
        :param filename:
            The filename to save the Spectrum1D object to.

        :type filename:
            str

        :param clobber: [optional]
            Whether to overwrite the `filename` if it already exists.

        :type clobber:
            bool

        :raises IOError:
            If the filename exists and we were not asked to clobber it.
        """
        
        if os.path.exists(filename) and not clobber:
            raise IOError("filename '{0}' exists and we have been asked not to"\
                " clobber it".format(filename))
        
        if not filename.endswith('fits'):
            # ASCII
            data = np.hstack([
                self.disp.reshape(-1, 1),
                self.flux.reshape(-1, 1),
                self.variance.reshape(-1, 1)
                ])
            return np.savetxt(filename, data, **kwargs)
            
        else:          
            # Create a tabular FITS format
            disp = pyfits.Column(name='disp', format='1D', array=self.disp)
            flux = pyfits.Column(name='flux', format='1D', array=self.flux)
            var = pyfits.Column(name='variance', format='1D', array=self.variance)
            table_hdu = pyfits.new_table([disp, flux, var])

            # Create Primary HDU
            hdu = pyfits.PrimaryHDU()

            # Update primary HDU with headers
            for key, value in self.headers.iteritems():
                if len(key) > 8: # To deal with ESO compatibility
                    hdu.header.update('HIERARCH {}'.format(key), value)

                try:
                    hdu.header.update(key, value)
                except ValueError:
                    logger.warn("Could not save header key/value combination: "\
                        "{0} = {1}".format(key, value))

            # Create HDU list with our tables
            hdulist = pyfits.HDUList([hdu, table_hdu])
            return hdulist.writeto(filename, clobber=clobber, **kwargs)


def cross_correlate_multiple(template_dispersion, template_fluxes,
    observed_spectrum, continuum_order=3, apodize=0.10):

    if template_dispersion.shape[0] != template_fluxes.shape[1]:
        raise ValueError("template dispersion must have size (N_pixels,) and "\
            "template fluxes must have size (N_models, N_pixels)")

    if not isinstance(observed_spectrum, Spectrum1D):
        raise TypeError("observed spectrum must be a Spectrum1D object")

    try:
        continuum_order = int(continuum_order)
    except (TypeError, ValueError):
        raise TypeError("continuum order must be an integer-like object")

    assert 1 > apodize >= 0, "Apodisation fraction must be between 0 and 1"
    

    N = template_dispersion.size
    N = N - 1 if N % 2 > 0 else N
    N_models = template_fluxes.shape[0]

    dispersion = template_dispersion[:N]
    template_flux = template_fluxes[:, :N]

    observed_flux = np.interp(dispersion, observed_spectrum.disp,
        observed_spectrum.flux, left=np.nan, right=np.nan)

    non_finite = ~np.isfinite(observed_flux)
    observed_flux[non_finite] = np.interp(dispersion[non_finite],
        dispersion[~non_finite], observed_flux[~non_finite])

    # Normalise
    if continuum_order >= 0:
        coeffs = np.polyfit(dispersion, observed_flux, continuum_order)
        observed_flux /= np.polyval(coeffs, dispersion)

    # Scale the flux level to that the template intensities
    observed_flux = (observed_flux * template_flux.ptp()) + template_flux.min()

    # Apodize edges
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
    template_flux_corr /= np.sqrt(np.inner(apod_observed_flux, apod_observed_flux))

    z_array = np.array(dispersion.copy())/dispersion[N/2] - 1.0

    z = np.ones(N_models) * np.nan
    z_err = np.ones(N_models) * np.nan
    R = np.ones(N_models) * np.nan
    for i in xrange(N_models):

        denominator = np.sqrt(np.inner(apod_template_flux[i, :], apod_template_flux[i, :]))
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

        best = z_array[ccf.argmax()]    
        err = (np.ptp(z_array[np.where(ccf >= 0.5*h)])/2.35482)**2

        z[i] = best
        z_err[i] = err
        R[i] = h

    return (z, z_err, R)


def cross_correlate(template, observed):
    """
    Return the measured redshift and uncertainty by cross-correlating the
    observed spectrum with the template (rest-frame) spectrum.

    :param template:
        The template spectrum, which is assumed to be at rest.

    :type template:
        :class:`specutils.Spectrum1D`

    :param observed:
        The observed spectrum.

    :type observed:
        :class:`specutils.Spectrum1D`

    :returns:
        The measured redshift and associated uncertainty.

    :rtype:
        tuple
    """
   
    if not isinstance(template, Spectrum1D):
        raise TypeError("template spectrum must be a Spectrum1D class")

    if not isinstance(observed, Spectrum1D):
        raise TypeError("observed spectrum must be a Spectrum1D class")

    # Put them all onto the same dispersion map
    N = observed.disp.size
    N = N - 1 if N % 2 > 0 else N

    dispersion = observed.disp[:N]
    assert np.isfinite(dispersion).all(), "All dispersion points must be finite"

    observed_flux = observed.flux[:N]
    template_flux = np.interp(dispersion, template.disp[:N], template.flux[:N],
        left=1, right=1)
    
    observed_flux[~np.isfinite(observed_flux)] = 1.
    template_flux[~np.isfinite(template_flux)] = 1.
    observed_flux /= observed_flux.mean()
    template_flux /= template_flux.mean()

    # Set up z array
    m = len(dispersion) / 2
    z_array = dispersion/dispersion[N/2] - 1.0
    
    # Apodize edges
    edge_buffer = 0.1 * (dispersion[-1] - dispersion[0])
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
    template_flux_corr = (fft_observed_flux * fft_template_flux.conjugate()) \
        / np.sqrt(np.inner(apod_observed_flux, apod_observed_flux) \
            * np.inner(apod_template_flux, apod_template_flux))

    correlation = np.fft.ifft(template_flux_corr).real

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

    return (z_best, z_err, h)
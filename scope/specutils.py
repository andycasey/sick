# coding: utf-8

""" Spectroscopy-related functionality """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["Spectrum1D", "Spectrum"]

# Standard library
import logging
import os

# Third-party
import numpy as np
import pyfits

from scipy import interpolate, ndimage
from scipy.optimize import leastsq

logger = logging.getLogger(__name__.split(".")[0])

# The following line of code will be supported until the end of the universe.
speed_of_light = 299792458e-3 # km/s

class Spectrum(object):
    """A class to deal with loading lots of different types of spectra"""

    @classmethod
    def load(cls, filename, **kwargs):

        # Try as a Spectrum1D class first
        methods = (Spectrum1D.load, load_aaomega_multispec)

        for method in methods:
            try:
                spectra = method(filename)
            except:
                continue

            else:
                if isinstance(spectra, Spectrum1D) and spectra.uncertainty is None:
                    spectra.uncertainty = np.array([0.002] * len(spectra.disp))

                return spectra

        raise IOError("could not interpret spectrum in {0}".format(filename))


class Spectrum1D(object):
    """This is a temporary class holder for a Spectrum1D object until the
    astropy.specutils.Spectrum1D module has advanced sufficiently to replace it."""
    
    headers = {}
    uncertainty = None

    def __init__(self, disp, flux, uncertainty=None, headers=None):
        """Initializes a `Spectrum1D` object with the given dispersion and flux
        arrays.
        
        Inputs
        ------
        disp : `np.array`
            Dispersion of the spectrum (i.e. the wavelength points).
            
        flux : `np.array`
            Flux points for each `disp` point.

        uncertainty : `np.array`
            Uncertainty in flux points for each dispersion point.
        """

        if len(disp) != len(flux):
            raise ValueError("dispersion and flux must have the same length")

        if len(disp) == 0:
            raise ValueError("dispersion and flux cannot be empty arrays")
        
        self.disp = disp
        self.flux = flux
        self.uncertainty = uncertainty
        if headers is not None:
            self.headers = headers

        return None

    def copy(self):
        """ Creates a copy of the object """

        return self.__class__(self.disp.copy(), self.flux.copy(),
            uncertainty=self.uncertainty, headers=self.headers)
    
    @classmethod
    def load(cls, filename, **kwargs):
        """Load a Spectrum1D from a given filename.
        
        Inputs
        ------
        filename : str
            Path of the filename to load. Can be either simple FITS extension
            or an ASCII filename.
            
        Notes
        ----
        If you are loading from an non-standard ASCII file, you can pass
        kwargs to `np.loadtxt` through this function.
        """
        
        if not os.path.exists(filename):
            raise IOError("Filename '%s' does not exist." % (filename, ))
        
        uncertainty = None

        if filename.endswith('.fits'):
            image = pyfits.open(filename, **kwargs)
            
            header = image[0].header
            
            # Check for a tabular data structure
            if len(image) > 1 and image[0].data is None:

                names = [name.lower() for name in image[1].data.names]
                dispersion_key = 'wave' if 'wave' in names else 'disp'
                
                disp, flux = image[1].data[dispersion_key], image[1].data['flux']

                if 'error' in names or 'uncertainty' in names:
                    uncertainty_key = 'error' if 'error' in names else 'uncertainty'

                    uncertainty = image[1].data[uncertainty_key]

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
            disp, flux = np.loadtxt(filename, unpack=True, **kwargs)
            
        return cls(disp, flux, uncertainty=uncertainty, headers=headers)


    def save(self, filename, clobber=True):
        """Saves the `Spectrum1D` object to the specified filename.
        
        Inputs
        ------
        filename : str
            The filename to save the `Spectrum1D` object to.
            
        clobber : bool, optional
            Whether to overwite the `filename` if it already exists.
        
        Raises
        ----
        IOError
            If the filename exists and we are not asked to clobber it.
            
        ValueError
            If the ``Spectrum1D`` object does not have a linear dispersion map.
        """
        
        if os.path.exists(filename) and not clobber:
            raise IOError("Filename '%s' already exists and we have been asked not to clobber it." % (filename, ))
        
        if not filename.endswith('fits'):
            # ASCII
            
            data = np.hstack([self.disp.reshape(len(self.disp), 1), self.flux.reshape(len(self.disp), 1)])
            
            assert len(data.shape) == 2
            assert data.shape[1] == 2
            
            np.savetxt(filename, data)
            
        else:
            # FITS
            crpix1, crval1 = 1, self.disp.min()
            
            cdelt1 = np.mean(np.diff(self.disp))
            
            test_disp = (crval1 + np.arange(len(self.disp), dtype=self.disp.dtype) * cdelt1).astype(self.disp.dtype)
            
            if np.max(self.disp - test_disp) > 10e-2 or self.uncertainty is not None:

                # Non-linear dispersion map, or we have uncertainty information too
                # Create a tabular FITS format.

                col_disp = pyfits.Column(name='disp', format='1D', array=self.disp)
                col_flux = pyfits.Column(name='flux', format='1D', array=self.flux)

                if self.uncertainty is not None:
                    col_uncertainty = pyfits.Column(name='uncertainty', format='1D', array=self.uncertainty)

                    table_hdu = pyfits.new_table([col_disp, col_flux, col_uncertainty])

                else:
                    table_hdu = pyfits.new_table([col_disp, col_flux])

                # Create Primary HDU
                hdu = pyfits.PrimaryHDU()

                # Update primary HDU with headers
                for key, value in self.headers.iteritems():
                    if len(key) > 8:
                        # To deal with ESO compatibility
                        hdu.header.update('HIERARCH %s' % (key, ), value)
                    
                    try:
                        hdu.header.update(key, value)

                    except ValueError:
                        logger.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                    
                # Create HDU list with our tables
                hdulist = pyfits.HDUList([hdu, table_hdu])

                hdulist.writeto(filename, clobber=clobber)

            else:
                # Linear dispersion map.
                # Create a PrimaryHDU file.

                # Ensure we have an array!
                hdu = pyfits.PrimaryHDU(np.array(self.flux))
                
                headers = self.headers.copy()
                headers.update({
                    'CRVAL1': crval1,
                    'CRPIX1': crpix1,
                    'CDELT1': cdelt1
                })
                
                for key, value in headers.iteritems():
                    if len(key) > 8:
                        # To deal with ESO compatibility
                        hdu.header.update('HIERARCH %s' % (key, ), value)
                    
                    else:
                        try:
                            hdu.header.update(key, value)

                        except ValueError:
                            logger.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                
                hdu.writeto(filename, clobber=clobber)
    

    def gaussian_smooth(self, fwhm, **kwargs):
        """ Convolves the spectrum flux with a Gaussian kernel.

        Inputs
        ------
        fwhm : float
            The FWHM of the Gaussian kernel to smooth with (Angstroms).
        """

        profile_sigma = abs(fwhm) / (2 * (2*np.log(2))**0.5)
        
        # The requested FWHM is in Angstroms, but the dispersion between each
        # pixel is likely less than an Angstrom, so we must calculate the true
        # smoothing value
        true_profile_sigma = profile_sigma / np.mean(np.diff(self.disp))
        smoothed_flux = ndimage.gaussian_filter1d(self.flux, true_profile_sigma, **kwargs)
        
        return self.__class__(self.disp, smoothed_flux, uncertainty=self.uncertainty,
            headers=self.headers)
        

    def doppler_shift(self, v):
        """Performs a Doppler correction on the given `Spectrum1D` object by the
        amount required.
        
        Inputs
        ------
        v : float
            The velocity (in km/s) to correct the `Spectrum1D` object by.
        """
        
        # Relatavistic:
        c = speed_of_light
        new_disp = self.disp * np.sqrt((1 + v/c)/(1 - v/c))
        return self.__class__(new_disp, self.flux, uncertainty=self.uncertainty, headers=self.headers)

    
    def interpolate(self, new_disp, mode='linear', bounds_error=False,
        fill_value=np.nan):
        """Interpolate the `Spectrum1D` onto a new dispersion map.
        
        Inputs
        ------
        new_disp : np.array
            An array of floating-point types containing the new dispersion points.
            
        mode : str
            Interpolation mode. See `scipy.interpolate.interp1d` for available
            options.
        
        bounds_error : bool
            See `scipy.interpolate.interp1d` for details.
        
        fill_value : float-type
            See `scipy.interpolate.interp1d`
        """
        
        f = interpolate.interp1d(self.disp, self.flux, kind=mode, copy=False,
                bounds_error=bounds_error, fill_value=fill_value)

        return self.__class__(new_disp, f(new_disp), uncertainty=self.uncertainty,
            headers=self.headers)


    def cross_correlate(self, template, wl_region=None, full_output=False):
        """Performs a cross-correlation between the observed and template spectrum and
        provides a radial velocity and associated uncertainty.

        Inputs
        ------
        template : `Spectrum1D`
            The normalised template spectrum.

        wl_region : optional, two length list containing floats [start, end]
            The starting and end wavelength to perform the cross-correlation on.

        full_output : `bool`, default False
            Whether or not to return the full output of the cross-correlation. If set to True
            then the output is as follows:

            v_rad, v_err, fft, profile

            where fft is a `np.ndarray` of shape (2, *) containing the Fourier transform
            and profile is a length 3 list containing the central peak point, peak height, and
            standard deviation.
        """

        if not isinstance(template, Spectrum1D):
            raise TypeError("template spectrum must be a `specutils.Spectrum1D` object")

        if wl_region is not None:

            if not isinstance(wl_region, (tuple, list, np.ndarray)) or len(wl_region) != 2:
                raise TypeError("wavelength region must be a two length list-type")

            try:
                wl_region = map(float, wl_region)

            except:
                raise TypeError("wavelength regions must be float-like")

        else:
            # Get overlapping region
            wl_region = np.array([
                np.max([self.disp[0], template.disp[0]]),
                np.min([self.disp[-1], template.disp[-1]])
                ])
        
        # Splice the observed spectrum
        idx = np.searchsorted(self.disp, wl_region)
        finite_values = np.isfinite(self.flux[idx[0]:idx[1]])

        observed_slice = Spectrum1D(disp=self.disp[idx[0]:idx[1]][finite_values], flux=self.flux[idx[0]:idx[1]][finite_values])

        # Ensure the template and observed spectra are on the same scale
        template_func = interpolate.interp1d(template.disp, template.flux, bounds_error=False, fill_value=0.0)
        template_slice = Spectrum1D(disp=observed_slice.disp, flux=template_func(observed_slice.disp))

        # Perform the cross-correlation
        padding = observed_slice.flux.size + template_slice.flux.size
        x_norm = (observed_slice.flux - observed_slice.flux[np.isfinite(observed_slice.flux)].mean(axis=None))
        y_norm = (template_slice.flux - template_slice.flux[np.isfinite(template_slice.flux)].mean(axis=None))

        Fx = np.fft.fft(x_norm, padding, )
        Fy = np.fft.fft(y_norm, padding, )
        iFxy = np.fft.ifft(Fx.conj() * Fy).real
        varxy = np.sqrt(np.inner(x_norm, x_norm) * np.inner(y_norm, y_norm))

        fft_result = iFxy/varxy

        # Put around symmetry
        num = len(fft_result) - 1 if len(fft_result) % 2 else len(fft_result)

        fft_y = np.zeros(num)

        fft_y[:num/2] = fft_result[num/2:num]
        fft_y[num/2:] = fft_result[:num/2]

        fft_x = np.arange(num) - num/2

        # Get initial guess of peak
        p0 = np.array([fft_x[np.argmax(fft_y)], np.max(fft_y), 10])

        gaussian_profile = lambda p, x: p[1] * np.exp(-(x - p[0])**2 / (2.0 * p[2]**2))
        errfunc = lambda p, x, y: y - gaussian_profile(p, x)

        try:
            p1, ier = leastsq(errfunc, p0.copy(), args=(fft_x, fft_y))

        except:
            raise

        # Uncertainty
        sigma = np.mean(2.0*(fft_y.real)**2)**0.5

        # Create functions for interpolating back onto the dispersion map
        points = (0, p1[0], sigma)
        interp_x = np.arange(num/2) - num/4

        functions = []
        for point in points:
            idx = np.searchsorted(interp_x, point)
            f = interpolate.interp1d(interp_x[idx-3:idx+3], observed_slice.disp[idx-3:idx+3], bounds_error=False, kind='cubic')
            
            functions.append(f)

        # 0, p1, sigma
        f, g, h = [func(point) for func, point in zip(functions, points)]

        # Calculate velocity 
        measured_vrad = speed_of_light * (1 - g/f)

        # Uncertainty
        measured_verr = np.abs(speed_of_light * (1 - h/f))
        R = np.max(fft_y)

        if full_output:
            results = [measured_vrad, measured_verr, R, np.vstack([fft_x, fft_y])]
            results.extend(p1)

            return results

        return [measured_vrad, measured_verr, R]


def load_aaomega_multispec(filename, fill_value=-1, clean=True):
    """
    Returns a list of Spectrum1D objects with headers from the main image
    and ones specific to that fibre (RA, DEC, X, Y, XERR, YERR, FIRE_NUM, etc)

    Inputs
    ------
    filename : str
        The reduced AAOmega multispec file to open.

    fill_value : float, optional
        A fill value to use for non-finite flux values.
    """
    
    image = pyfits.open(filename)
    
    req_image_headers = ['MEANRA', 'MEANDEC', 'EPOCH', 'EXPOSED', 'TOTALEXP', 'UTDATE',
        'UTSTART', 'UTEND', 'EXPOSED', 'ELAPSED', 'TOTALEXP', 'RO_GAIN', 'RO_NOISE', 'TELESCOP',
        'ALT_OBS', 'LAT_OBS', 'LONG_OBS', 'OBJECT' ]
    req_fibre_headers = ['NAME', 'RA', 'DEC', 'X', 'Y', 'XERR', 'YERR', 'MAGNITUDE', 'COMMENT']
    
    base_headers = {}
    for header in req_image_headers:
        try:
            base_headers[header] = image[0].header[header]
        except KeyError:
            logger.info('Could not find "{keyword}" keyword in the headers of filename {filename}'
                .format(keyword=header, filename=filename))
    
    dispersion = image[0].header['CRVAL1'] \
        + (np.arange(image[0].header['NAXIS1']) - image[0].header['CRPIX1']) * image[0].header['CDELT1']
    
    spectra = []    
    columns = image[2].columns.names

    program_indices = np.where(image[2].data["TYPE"] == "P")[0]
    
    for i, index in enumerate(program_indices):
    
        headers = base_headers.copy()
        headers['FIBRE_NUM'] = i + 1
        
        for header in req_fibre_headers:
            headers[header] = image[2].data[index][header]
        
        flux = np.array(image[0].data[index], dtype=np.float)
        uncertainty = np.sqrt(abs(flux))
        #uncertainty = np.abs(np.random.normal(0, np.sqrt(np.median(np.abs(flux))), size=len(flux)))

        # Remove 1 pixel from each where there is a nan
        if clean:
            non_finite_diffs = np.where(np.diff(np.array(np.isfinite(flux), dtype=int)) > 0)[0]
            # Set each pixel +/- 1 of the non_finite_diffs as non_finite
            for pixel in non_finite_diffs:
                flux[pixel - 1: pixel + 2] = np.nan

        # Check if it's worthwhile having these
        if all(~np.isfinite(flux)):
            flux = np.array([fill_value] * len(flux), dtype=np.float)

        flux[0 >= flux] = np.nan

        spectrum = Spectrum1D(dispersion, flux, uncertainty=uncertainty, headers=headers)
        spectra.append(spectrum)
    
    return spectra

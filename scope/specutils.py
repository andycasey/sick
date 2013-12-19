# coding: utf-8

""" A basic Spectrum class for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os

# Third-party
import pyfits
import numpy as np

from scipy import interpolate, ndimage, polyfit, poly1d
from scipy.optimize import leastsq

__all__ = ['Spectrum1D', 'aat_aaomega']

# The following line of code will be supported until the end of the universe.
speed_of_light = 299792458e-3 # km/s


class Spectrum1D(object):
    """This is a temporary class holder for a Spectrum1D object until the
    astropy.specutils.Spectrum1D module has advanced sufficiently to replace it."""
    
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

        if 0 in map(len, [disp, flux]):
            raise ValueError("dispersion and flux cannot be empty arrays")
        
        self.disp = disp
        self.flux = flux
        self.uncertainty = uncertainty
        self.headers = headers
        
        return None
    
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

                if np.all([header.has_key(key) for key in ('CDELT1', 'NAXIS1', 'CRVAL1')]):
                    disp = header['CRVAL1'] + np.arange(header['NAXIS1']) * header['CDELT1']
            
                if header.has_key('LTV1'):
                    disp -= header['LTV1'] * header['CDELT1']

                #disp -= header['LTV1'] if header.has_key('LTV1') else 0
                flux = image[0].data
            

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
                
                if headers.has_key(key):
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
                        logging.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                    
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
                            logging.warn("Could not save header key/value combination: %s = %s" % (key, value, ))
                
                hdu.writeto(filename, clobber=clobber)
    

    def gaussian_smooth(self, fwhm, **kwargs):
        """ Convolves the spectrum flux with a Gaussian kernel.

        Inputs
        ------
        fwhm : float
            The FWHM of the Gaussian kernel to smooth with.
        """



        profile_sigma = abs(fwhm) / (2 * (2*np.log(2))**0.5)
        
        # The requested FWHM is in Angstroms, but the dispersion between each
        # pixel is likely less than an Angstrom, so we must calculate the true
        # smoothing value
        
        true_profile_sigma = profile_sigma / np.mean(np.diff(self.disp))
        smoothed_flux = ndimage.gaussian_filter1d(self.flux, true_profile_sigma, **kwargs)
        
        return self.__class__(
            self.disp,
            smoothed_flux,
            uncertainty=self.uncertainty,
            headers=self.headers)
        

    def doppler_shift(self, velocity):
        """Performs a Doppler correction on the given `Spectrum1D` object by the
        amount required.
        
        Inputs
        ------
        velocity : float
            The velocity (in km/s) to correct the `Spectrum1D` object by.
        """
        
        new_disp = self.disp * np.sqrt((1 + velocity/speed_of_light)/(1 - velocity/speed_of_light))
        #new_disp = (self.disp * (1 + velocity/speed_of_light))/np.sqrt(1 - velocity**2/speed_of_light**2)
            
        return self.__class__(
            new_disp,
            self.flux,
            uncertainty=self.uncertainty,
            headers=self.headers)

    
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

        if self.uncertainty is not None:
            # Probably not the 'right' thing to do, but to a first approximation..
            f_u = interpolate.interp1d(self.disp, self.uncertainty, kind=mode, copy=False,
                                       bounds_error=bounds_error, fill_value=fill_value)

            uncertainty = f_u(new_disp)

        else:
            uncertainty = None

        return self.__class__(
            new_disp,
            f(new_disp),
            uncertainty=uncertainty,
            headers=self.headers
            )


    def cross_correlate(self, template, wl_region, full_output=False):
        """Performs a cross-correlation between the observed and template spectrum and
        provides a radial velocity and associated uncertainty.

        Inputs
        ------
        template : `Spectrum1D`
            The normalised template spectrum.

        wl_region : two length list containing floats [start, end]
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

        if not isinstance(wl_region, (tuple, list, np.ndarray)) or len(wl_region) != 2:
            raise TypeError("wavelength region must be a two length list-type")

        try:
            wl_region = map(float, wl_region)

        except:
            raise TypeError("wavelength regions must be float-like")

        
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
            f = interpolate.interp1d(interp_x[idx-3:idx+3], observed_slice.disp[idx-3:idx+3], bounds_error=True, kind='cubic')

            functions.append(f)

        # 0, p1, sigma
        f, g, h = [func(point) for func, point in zip(functions, points)]


        # Calculate velocity 
        measured_vrad = speed_of_light * (1 - g/f)

        # Uncertainty
        measured_verr = np.abs(speed_of_light * (1 - h/f))

        
        if full_output:
            results = [measured_vrad, measured_verr, np.vstack([fft_x, fft_y])]
            results.extend(p1)

            return results

        return [measured_vrad, measured_verr]


    def fit_continuum(self, knot_spacing=200, lower_clip=1.0, upper_clip=0.20, \
                      max_iterations=3, order=3, exclude=None, include=None, \
                      additional_points=None, function='spline', scale=1.0, **kwargs):
        """Fits the continuum for a given `Spectrum1D` spectrum.
        
        Inputs
        ------
        knot_spacing : float or None, optional
            The knot spacing for the continuum spline function in Angstroms. Optional.
            If not provided then the knot spacing will be determined automatically.
        
        lower_clip : float, optional
            This is the lower sigma clipping level. Optional.

        upper_clip : float, optional
            This is the upper sigma clipping level. Optional.
            
        max_iterations : int, optional
            Maximum number of spline-fitting operations.
            
        order : int, optional
            The order of the spline function to fit.
            
        exclude : list of tuple-types containing floats, optional
            A list of wavelength regions to always exclude when determining the
            continuum. Example:
            
            >> exclude = [
            >>    (3890.0, 4110.0),
            >>    (4310.0, 4340.0)
            >>  ]
            
            In the example above the regions between 3890 A and 4110 A, as well as
            4310 A to 4340 A will always be excluded when determining the continuum
            regions.

        function: only 'spline' or 'poly'

        scale : float
            A positive scaling factor to apply to the normalised flux levels.
            
        include : list of tuple-types containing floats, optional
            A list of wavelength regions to always include when determining the
            continuum.
        """

        logging.debug("fit_continuum({self}, {function}, {knot_spacing}, {sigma_clip}, {iter}, {order}, {scale})".format(self=self,function=function,knot_spacing=knot_spacing,sigma_clip=(lower_clip,upper_clip), iter=max_iterations,order=order,scale=scale))
        
        scale = abs(scale)

        exclusions = []
        continuum_indices = range(len(self.flux))

        # Snip left and right
        finite_positive_flux = np.isfinite(self.flux) * self.flux > 0

        #print "finite flux", np.any(finite_positive_flux), finite_positive_flux
        #print "where flux", np.where(finite_positive_flux)
        #print "flux is...", self.flux
        left_index = np.where(finite_positive_flux)[0][0]
        right_index = np.where(finite_positive_flux)[0][-1]

        # See if there are any regions we need to exclude
        if exclude is not None and len(exclude) > 0:
            exclude_indices = []
            
            if isinstance(exclude[0], float) and len(exclude) == 2:
                # Only two floats given, so we only have one region to exclude
                exclude_indices.extend(range(*np.searchsorted(self.disp, exclude)))
                
            else:
                # Multiple regions provided
                for exclude_region in exclude:
                    exclude_indices.extend(range(*np.searchsorted(self.disp, exclude_region)))
        
            continuum_indices = np.sort(list(set(continuum_indices).difference(exclude_indices)))
        
        # See if there are any regions we should always include
        if include is not None and len(include) > 0:
            include_indices = []
            
            if isinstance(include[0], float) and len(include) == 2:
                # Only two floats given, so we can only have one region to include
                include_indices.extend(range(*np.searchsorted(self.disp, include)))
                
            else:
                # Multiple regions provided
                for include_region in include:
                    include_indices.extend(range(*np.searchsorted(self.disp, include_region)))
        

        # We should exclude non-finite numbers from the fit
        non_finite_indices = np.where(~np.isfinite(self.flux))[0]
        continuum_indices = np.sort(list(set(continuum_indices).difference(non_finite_indices)))

        # We should also exclude zero or negative flux points from the fit
        zero_flux_indices = np.where(0 >= self.flux)[0]
        continuum_indices = np.sort(list(set(continuum_indices).difference(zero_flux_indices)))

        if knot_spacing is None or knot_spacing == 0:
            knots = []

        else:
            knot_spacing = abs(knot_spacing)
            
            end_spacing = ((self.disp[-1] - self.disp[0]) % knot_spacing) /2.
        
            if knot_spacing/2. > end_spacing: end_spacing += knot_spacing/2.
                
            knots = np.arange(self.disp[0] + end_spacing, self.disp[-1] - end_spacing + knot_spacing, knot_spacing)
            if len(knots) > 0 and knots[-1] > self.disp[continuum_indices][-1]:
                knots = knots[:knots.searchsorted(self.disp[continuum_indices][-1])]
                
            if len(knots) > 0 and knots[0] < self.disp[continuum_indices][0]:
                knots = knots[knots.searchsorted(self.disp[continuum_indices][0]):]


        for iteration in xrange(max_iterations):
            
            splrep_disp = self.disp[continuum_indices]
            splrep_flux = self.flux[continuum_indices]

            splrep_weights = np.ones(len(splrep_disp))

            # We need to add in additional points at the last minute here
            if additional_points is not None and len(additional_points) > 0:

                for point, flux, weight in additional_points:

                    # Get the index of the fit
                    insert_index = np.searchsorted(splrep_disp, point)
                    
                    # Insert the values
                    splrep_disp = np.insert(splrep_disp, insert_index, point)
                    splrep_flux = np.insert(splrep_flux, insert_index, flux)
                    splrep_weights = np.insert(splrep_weights, insert_index, weight)



            if function == 'spline':
                order = 5 if order > 5 else order
                

                tck = interpolate.splrep(splrep_disp, \
                                         splrep_flux, \
                                         k=order, task=-1, t=knots, w=splrep_weights)

                continuum = interpolate.splev(self.disp, tck)

            elif function == 'poly':
            
                p = poly1d(polyfit(splrep_disp, splrep_flux, order))
                continuum = p(self.disp)

            else:
                raise ValueError("Unknown function type: only spline or poly available")
            

            difference = continuum - self.flux
        
            #n = 100./np.median(np.diff(self.disp))
            #n = 100
            #convolved_flux = np.convolve(self.flux, np.ones(n)/n)
            #sigma_difference = (difference - np.median(difference)) / np.std(difference)
            sigma_difference = difference / np.std(difference)

            # Clip 
            upper_exclude = np.where(sigma_difference > abs(upper_clip))[0]
            lower_exclude = np.where(sigma_difference < -abs(lower_clip))[0]
            
            exclude_indices = list(upper_exclude)
            exclude_indices.extend(lower_exclude)
            exclude_indices = np.array(exclude_indices)
            
            if len(exclude_indices) is 0: break
            
            exclusions.extend(exclude_indices)
            
            # Before excluding anything, we must check to see if there are regions
            # which we should never exclude
            if include is not None:
                exclude_indices = set(exclude_indices).difference(include_indices)
            
            # Remove regions that have been excluded
            continuum_indices = np.sort(list(set(continuum_indices).difference(exclude_indices)))
        
        #return self.__class__(disp=self.disp, flux=continuum, headers=self.headers)
        
        # Snip the edges based on exclude regions
        if exclude is not None:

            for exclude_region in exclude:

                start, end = np.searchsorted(self.disp, exclude_region)

                if end >= right_index > start:
                    # Snip the edge
                    right_index = start

                if end > left_index >= start:
                    # Snip the edge
                    left_index = end

        # Apply flux scaling
        continuum *= scale

        normalised = self.__class__(
            disp=self.disp[left_index:right_index],
            flux=(self.flux/continuum)[left_index:right_index],
            uncertainty=self.uncertainty[left_index:right_index] if self.uncertainty is not None else None,
            headers=self.headers)

        continuum = self.__class__(disp=self.disp[left_index:right_index], flux=continuum[left_index:right_index])

        return (normalised, continuum)



def load_aaomega_multispec(filename, fill_value=0):
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
    
    req_image_headers = ['MEANRA', 'MEANDEC', 'DATE', 'EPOCH', 'EXPOSED', 'TOTALEXP', 'UTDATE',
        'UTSTART', 'UTEND', 'EXPOSED', 'ELAPSED', 'TOTALEXP', 'RO_GAIN', 'RO_NOISE', 'TELESCOP',
        'ALT_OBS', 'LAT_OBS', 'LONG_OBS', 'OBJECT' ]
    req_fibre_headers = ['NAME', 'RA', 'DEC', 'X', 'Y', 'XERR', 'YERR', 'MAGNITUDE', 'COMMENT']
    
    base_headers = {}
    for header in req_image_headers:
        try:
            base_headers[header] = image[0].header[header]
        except KeyError:
            logging.info('Could not find "{keyword}" keyword in the headers of filename {filename}'
                .format(keyword=header, filename=filename))
    
    dispersion = image[0].header['CRVAL1'] \
        + (np.arange(image[0].header['NAXIS1']) - image[0].header['CRPIX1']) * image[0].header['CDELT1']
    
    spectra = []    
    columns = image[2].columns.names

    for i, star in enumerate(image[2].data):
        
        if star['TYPE'] == 'P': # Program object
            
            headers = base_headers.copy()
            headers['FIBRE_NUM'] = i + 1
            
            for header in req_fibre_headers:
                headers[header] = star[header]
            
            flux = image[0].data[i]
            
            # Check if it's worthwhile having these
            if all(~np.isfinite(flux)):
                flux = np.array([fill_value] * len(flux))

            # Remove off the edge nan's                
            left_side = list(np.isfinite(flux)).index(True)
            right_side = -list(np.isfinite(flux[::-1])).index(True)

            dispersion_copy = dispersion.copy()
            dispersion_copy = dispersion_copy[left_side:right_side]
            flux = flux[left_side:right_side]
            
            #if len(dispersion_copy) == 0: continue

            # Now fill any remaining values
            remaining_nans = ~np.isfinite(flux)
            flux[remaining_nans] = fill_value
            
            if len(flux) == 0:
                spectra.append(None)
            else:
                spectrum = Spectrum1D(dispersion_copy, flux, headers=headers)
                spectra.append(spectrum)
    
    return spectra

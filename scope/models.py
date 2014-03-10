# coding: utf-8

""" Handles the loading and interpolation of flux models for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging
import os
import re
from glob import glob

# Third-party
import numpy as np
import pyfits
from scipy import interpolate, ndimage

# Module
from utils import human_readable_digit
from specutils import Spectrum1D


__all__ = ['Model', 'load_model_data']

class Model(object):
    """ Model class for SCOPE """

    def __init__(self, configuration, internal_cache=False):
        self.configuration = configuration

        # Dispersions
        self.dispersion = {}
        for beam, dispersion_filename in configuration['models']['dispersion_filenames'].iteritems():
            self.dispersion[beam] = load_model_data(dispersion_filename)

        grid_points = {}
        flux_filenames = {}

        # Read the points from filenames
        for beam in configuration['models']['flux_filenames']:
            folder = configuration['models']['flux_filenames'][beam]['folder']
            re_match = configuration['models']['flux_filenames'][beam]['re_match']

            all_filenames = glob(os.path.join(folder, '*'))

            points = []
            matched_filenames = []
            for filename in all_filenames:
                match = re.match(re_match, os.path.basename(filename))

                if match is not None:
                    if not hasattr(self, 'colnames'):
                        colnames = []
                        groups = match.groups()

                        groupdict = match.groupdict()
                        for value in match.groupdict().itervalues():
                            if groups.count(value) > 1: break
                            colnames.append(match.groupdict().keys()[groups.index(value)])

                        if len(colnames) == len(groups):
                            self.colnames = colnames

                    points.append(map(float, match.groups()))
                    matched_filenames.append(filename)

            grid_points[beam] = points
            flux_filenames[beam] = matched_filenames

        first_beam = configuration['models']['flux_filenames'].keys()[0]
        self.grid_points = np.array(grid_points[first_beam])

        self.grid_boundaries = {}
        for i, colname in enumerate(self.colnames):
            self.grid_boundaries[colname] = np.array([
                np.min(self.grid_points[:, i]),
                np.max(self.grid_points[:, i])
                ])

        #dtype = np.dtype({'names': tuple(self.colnames), 'formats': tuple(['<f8'] * len(self.colnames))})
        #self.grid_points = np.array(self.grid_points, dtype=dtype)
        #self.flux_cache = {}

        # If it's just the one beam, it's easy!        
        if len(configuration['models']['dispersion_filenames'].keys()) == 1:
            self.flux_filenames = flux_filenames[first_beam]
            return None
        #    if internal_cache:
        #        self.flux_cache[first_beam] = np.zeros((len(self.flux_filenames), 
        #                len(load_model_data(self.flux_filenames[0]))))
        #        self.flux_cache[first_beam][:] = np.nan

        else:
            self.flux_filenames = {first_beam: flux_filenames[first_beam]}

        #    if internal_cache:
        #        self.flux_cache[first_beam] = np.zeros((len(self.flux_filenames[first_beam]), 
        #                len(load_model_data(self.flux_filenames[first_beam][0]))))
        #        self.flux_cache[first_beam][:] = np.nan
            

        # Put all points and filenames on the one scale
        for beam in configuration['models']['dispersion_filenames'].keys()[1:]:
            
            points = grid_points[beam]
            if len(points) != len(self.grid_points):
                raise ValueError("number of model points found in {first_beam} beam ({num_first_beam})"
                    " did not match the number in {this_beam} beam ({num_this_beam})"
                    .format(first_beam=first_beam, num_first_beam=len(self.grid_points), this_beam=beam,
                        num_this_beam=len(points)))

            sort_indices = []

            for point in points:
                index = self.check_grid_point(point)
                sort_indices.append(index)

            self.flux_filenames[beam] = [flux_filenames[beam][index] for index in sort_indices]

        #    if internal_cache:
        #        self.flux_cache[beam] = np.zeros((len(self.flux_filenames[beam]), 
        #            len(load_model_data(self.flux_filenames[beam][0]))))
        #        self.flux_cache[beam][:] = np.nan

        return None


    def __repr__(self):
        num_apertures = len(self.dispersion)
        num_models = len(self.grid_points) * num_apertures
        num_pixels = sum([len(dispersion) * num_models for dispersion in self.dispersion.values()])
        
        return '{module}.Model({num_models} models, {num_apertures} apertures: "{apertures}", {num_parameters} parameters: "{parameters}", ~{num_pixels} pixels)'.format(
            module=self.__module__, num_models=num_models, num_apertures=num_apertures, apertures=', '.join(self.dispersion.keys()),
            num_pixels=human_readable_digit(num_pixels), num_parameters=self.grid_points.shape[1],
            parameters=', '.join(self.colnames))


    def get_nearest_neighbours(self, point, n=1):
        """Returns the indices for the nearest `n` neighbours to the given `point` in each dimension.

        Inputs
        ------
        point : list of `float` values
            The point to find neighbours for.

        n : int
            The number of neighbours to find either side of `point` in each dimension.
            Therefore the total number of points returned will be dim(point)^n.
        """

        if len(point) != self.grid_points.shape[1]:
            raise ValueError("point length ({length}) is incompatible with grid shape ({shape})"
                .format(length=len(point), shape=self.grid_points.shape))

        try: n = int(n)
        except TypeError:
            raise TypeError("number of neighbours must be an integer-type")
        if 1 > n:
            raise ValueError("number of neighbours must be a positive integer-type")

        indices = set(np.arange(len(self.grid_points)))
        for i, point_value in enumerate(point):
            difference = np.unique(self.grid_points[:, i] - point_value)

            if sum(difference > 0) * sum(difference < 0) == 0:
                return ValueError("point ({0}) outside of the grid boundaries".format(point_value))

            limit_min = difference[np.where(difference < 0)][-n:][0] + point_value
            limit_max = difference[np.where(difference > 0)][:n][-1] + point_value
    
            these_indices = np.where((limit_max >= self.grid_points[:, i]) & (self.grid_points[:, i] >= limit_min))[0]
            indices.intersection_update(these_indices)

        return np.array(list(indices))


    def check_grid_point(self, point):
        """Checks whether the point provided exists in the grid of models. If so,
        its index is returned.

        Inputs
        ------
        point : list of float values
            The point of interest.
        """

        index = np.where(np.all(np.equal(self.grid_points - point, np.zeros(len(point))), 1))[0]

        if len(index) > 0:
            return index[0]

        return False


    def interpolate_flux(self, point, beams='all', kind='linear', **kwargs):
        """Interpolates through the grid of models to the given `point` and returns
        the interpolated flux.

        Inputs
        ------
        point : list of `float` values
            The point to interpolate to.

        beams : str, optional
            The beams to interpolate flux for. If this is 'all', a dictionary is
            returned with interpolated fluxes for all beams.

        kind : str or int, optional
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
            or as an integer specifying the order of the spline interpolator
            to use. Default is 'linear'.
        """

        neighbour_indices = self.get_nearest_neighbours(point)

        if beams is 'all':
            beams = self.dispersion.keys()

        elif not isinstance(beams, (list, tuple)):
            beams = [beams]

        for beam in beams:
            if beam not in self.dispersion.keys():
                raise ValueError("could not find '{beam}' beam".format(beam=beam))

        interpolated_flux = {}
        for beam in beams:

            beam_flux = np.zeros((
                len(neighbour_indices),
                len(self.dispersion[beam])
                ))
            beam_flux[:] = np.nan

            # Load the flux points
            flux_data = self.flux_filenames if len(self.dispersion.keys()) == 1 else self.flux_filenames[beam]
            for i, index in enumerate(neighbour_indices):
                #if beam in self.flux_cache and not np.all(~np.isfinite(self.flux_cache[beam][index])):
                #    beam_flux[i, :] = self.flux_cache[beam][index]
                #    logger.info("CACHING")
                #else:
                #    self.flux_cache[beam][index] = load_model_data(flux_data[index])
                beam_flux[i, :] = load_model_data(flux_data[index])
            
            try:
                interpolated_flux[beam] = interpolate.griddata(
                    self.grid_points[neighbour_indices],
                    beam_flux,
                    [point],
                    **kwargs).flatten()

            except:
                # Return all nans!
                continue

        return interpolated_flux


    def map_apertures(self, observed_dispersions):
        """References model spectra to each observed spectra based on their wavelengths.

        Inputs
        ------
        observed_dispsersions : a list of observed dispersion maps (e.g. list-types full of floats)
            The dispersion maps for all observed apertures.
        """

        mapped_apertures = []
        for i, observed_dispersion in enumerate(observed_dispersions):

            # Initialise the list
            apertures_found = []

            observed_wlmin = np.min(observed_dispersion)
            observed_wlmax = np.max(observed_dispersion)

            for model_aperture, model_dispersion in self.dispersion.iteritems():

                model_wlmin = np.min(model_dispersion)
                model_wlmax = np.max(model_dispersion)

                # Is there overlap?
                if (model_wlmin < observed_wlmin and observed_wlmax < model_wlmax) \
                or (observed_wlmin < model_wlmin and model_wlmax < observed_wlmax) \
                or (model_wlmin < observed_wlmin and (observed_wlmin < model_wlmax and model_wlmax < observed_wlmax)) \
                or ((observed_wlmin < model_wlmin and model_wlmin < observed_wlmax) and observed_wlmax < model_wlmax):
                    apertures_found.append(model_aperture)

            if len(apertures_found) == 0:
                raise ValueError("no model apertures found for observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                    .format(wl_start=observed_wlmin, wl_end=observed_wlmax))

            elif len(apertures_found) > 1:
                raise ValueError("multiple model apertures found for observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                    .format(wl_start=observed_wlmin, wl_end=observed_wlmax))

            mapped_apertures.append(apertures_found[0])

        # Check that the mean pixel size in the model dispersion maps is smaller than the observed dispersion maps
        for aperture, observed_dispersion in zip(mapped_apertures, observed_dispersions):

            mean_observed_pixel_size = np.mean(np.diff(observed_dispersion))
            mean_model_pixel_size = np.mean(np.diff(self.dispersion[aperture]))
            if mean_model_pixel_size > mean_observed_pixel_size:
                raise ValueError("the mean model pixel size in the {aperture} aperture is larger than the mean"
                    " pixel size in the observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                    .format(aperture=aperture, wl_start=np.min(observed_dispersion), wl_end=np.max(observed_dispersion)))

        # Keep an internal reference of the aperture mapping
        self._mapped_apertures = mapped_apertures

        return mapped_apertures


    def model_spectra(self, observations=None, **kwargs):
        """ Interpolates flux values for a set of stellar parameters and
        applies any relevant smoothing or normalisation of the data. 

        Inputs
        ------
        parameters : list of floats
            The input parameters that were provdided to the `chi_squared_fn` function.

        parameter_names : list of str, should be same length as `parameters`.
            The names for the input parameters.
        """

        # Build the grid point
        grid_point = [kwargs[stellar_parameter] for stellar_parameter in self.colnames]

        # Interpolate the flux
        try:
            interpolated_flux = self.interpolate_flux(grid_point)

        except:
            logging.debug("No model flux could be determined for {0}".format(
                ", ".join(["{0} = {1:.2f}".format(parameter, value) \
                    for parameter, value in zip(self.colnames, grid_point)])))
            return None

        logging.debug("Interpolated model flux at {0}".format(
            ", ".join(["{0} = {1:.2f}".format(parameter, value) \
                for parameter, value in zip(self.colnames, grid_point)])))

        if interpolated_flux == {}: return None
        for aperture, flux in interpolated_flux.iteritems():
            if np.all(~np.isfinite(flux)): return None

        # Create spectra
        model_spectra = {}
        for aperture, interpolated_flux in interpolated_flux.iteritems():
            model_spectra[aperture] = Spectrum1D(
                disp=self.dispersion[aperture], flux=interpolated_flux)

        # Any synthetic smoothing to apply?
        for aperture in self._mapped_apertures:
            key = 'smooth_model_flux.{aperture}.kernel'.format(aperture=aperture)

            # Is the smoothing a free parameter?
            if key in kwargs:
                # Ensure valid smoothing value
                if kwargs[key] < 0: return None
                model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(kwargs[key])

            elif self.configuration['smooth_model_flux'][aperture]['perform']:
                # Apply a single smoothing value
                kernel = self.configuration['smooth_model_flux'][aperture]['kernel']
                model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(kernel)
                logging.debug("Smoothed model flux for '{0}' aperture with kernel {1}".format(
                    aperture, kernel))
        
        # Interpolate synthetic to observed dispersion map
        if observations is not None:
            for aperture, observed_spectrum in zip(self._mapped_apertures, observations):
                model_spectra[aperture] = model_spectra[aperture].interpolate(observed_spectrum.disp)

        return model_spectra


    def observed_spectra(self, observations, **kwargs):
        """ Prepares the observed spectra for comparison with model spectra
        by performing normalisation and doppler shifts.

        Inputs
        ------
        parameters : list of floats
            The input parameters that were provdided to the `chi_squared_fn` function.

        parameter_names : list of str, should be same length as `parameters`.
            The names for the input parameters.

        observations : list of `Spectrum1D` objects
            The observed spectra.
        """

        logging.debug("Preparing observed spectra for comparison...")

        modified_spectra = []
        for aperture, spectrum in zip(self._mapped_apertures, observations):
            
            modified_spectrum = spectrum.copy()

            # Any normalisation to perform?
            if self.configuration["normalise_observed"][aperture]["perform"]:
            
                # Since we need to perform normalisation, the normalisation coefficients
                # should be in kwargs
                num_coefficients_expected = self.configuration["normalise_observed"][aperture]["order"] + 1
                coefficients = [kwargs["normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n)] \
                    for n in xrange(num_coefficients_expected)]

                continuum = np.polyval(coefficients, modified_spectrum.disp)
                modified_spectrum.flux /= continuum

            # Any doppler shift to perform?
            if self.configuration["doppler_shift"][aperture]["perform"]:

                velocity = kwargs["doppler_shift.{0}".format(aperture)]
                modified_spectrum.doppler_shift(velocity)

            modified_spectra.append(modified_spectrum)

        return modified_spectra


    def masks(self, model_spectra):
        """Returns pixel masks to apply to the model spectra
        based on the configuration provided.

        Inputs
        ------
        model_spectra : dict
            A dictionary containing aperture names as keys and specutils.Spectrum1D objects
            as values.
        """

        if "masks" not in self.configuration:
            masks = {}
            for aperture, spectrum in model_spectra.iteritems():
                masks[aperture] = np.ones(len(spectrum.disp))

        else:
            masks = {}
            for aperture, spectrum in model_spectra.iteritems():
                if aperture not in self.configuration["masks"]:
                    masks[aperture] = np.ones(len(spectrum.disp))
                
                else:
                    # We are required to build a mask.
                    mask = np.zeros(len(spectrum.disp))
                    if self.configuration["masks"][aperture] is not None:
                        for region in self.configuration["masks"][aperture]:
                            index_start, index_end = np.searchsorted(spectrum.disp, region)
                            mask[index_start:index_end] = 1

                    masks[aperture] = mask

        return masks


    def weights(self, model_spectra):
        """Returns callable weighting functions to apply to the \chi^2 comparison.

        Inputs
        ------
        model_spectra : dict
            A dictionary containing aperture names as keys and specutils.Spectrum1D objects
            as values.
        """

        if "weights" not in self.configuration:
            weights = {}
            for aperture, spectrum in model_spectra.iteritems():
                # Equal weighting to all pixels
                weights[aperture] = lambda disp, flux: np.ones(len(flux))

        else:
            weights = {}
            for aperture, spectrum in model_spectra.iteritems():
                if aperture not in self.configuration["weights"]:
                    # Equal weighting to all pixels
                    weights[aperture] = lambda disp, flux: np.ones(len(flux))

                else:
                    # Evaluate the expression, providing numpy (as np), disp, and flux as locals
                    weights[aperture] = lambda disp, flux: eval(self.configuration["weights"][aperture], 
                        {"disp": disp, "np": np, "flux": flux})

        return weights



def load_model_data(filename, **kwargs):
    """Loads dispersion/flux values from a given filename. This can be either a 1-column ASCII
    file, or a single extension FITS file.

    Inputs
    ------
    filename : `str`
        The filename to load the values from.
    """

    # Check the open mode
    if filename.endswith('.fits'):
        with pyfits.open(filename, **kwargs) as image:
            data = image[0].data

    elif filename.endswith('.cached'):

        # Put in our preferred keyword arguments
        kwargs.setdefault('mode', 'r')
        kwargs.setdefault('dtype', np.float32)

        data = np.memmap(filename, **kwargs)
        if kwargs["mode"] == "r":
            # Wrapping the data as a new array so that it is pickleable across
            # multiprocessing without the need to share the memmap. See:
            # http://stackoverflow.com/questions/15986133/why-does-a-memmap-need-a-filename-when-i-try-to-reshape-a-numpy-array
            data = np.array(data)

    else:
        # Assume it must be ASCII.
        data = np.loadtxt(filename, **kwargs)

    return data



        
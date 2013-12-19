# coding: utf-8

""" Handles the loading and interpolation of flux models for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

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

__all__ = ['Models', 'load_model_data']

class Models(object):
    """Class for interpolating model fluxes for SCOPE."""

    def __init__(self, configuration):
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

        #dtype = np.dtype({'names': tuple(self.colnames), 'formats': tuple(['<f8'] * len(self.colnames))})
        #self.grid_points = np.array(self.grid_points, dtype=dtype)

        # If it's just the one beam, it's easy!        
        if len(configuration['models']['flux_filenames'].keys()) == 1:
            self.flux_filenames = flux_filenames[first_beam]

            return None

        else:
            self.flux_filenames = {first_beam: flux_filenames[first_beam]}

        # Put all points and filenames on the one scale
        for beam in configuration['models']['flux_filenames'].keys()[1:]:
            
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

        return None


    def __repr__(self):
        num_apertures = len(self.dispersion)
        num_models = len(self.grid_points) * num_apertures
        num_pixels = sum([len(dispersion) * num_models for dispersion in self.dispersion.values()])
        
        return 'Models({num_models} models, {num_apertures} apertures: "{apertures}", {num_parameters} parameters: "{parameters}", ~{num_pixels} pixels)'.format(
            num_models=num_models,
            num_apertures=num_apertures,
            apertures=', '.join(self.dispersion.keys()),
            num_pixels=human_readable_digit(num_pixels),
            num_parameters=self.grid_points.shape[1],
            parameters=', '.join(self.colnames))


    def pre_cache(self):
        """Perform pre-caching: smooth the model fluxes and interpolate them onto a dispersion map
        which is similar to the observations.
        """

        # Check if we have a pre-cache configuration
        if 'pre-cache' not in self.configuration:
            raise KeyError("no 'pre-cache' information found in the configuration")

        logging.debug("Beginning pre-caching...")

        # TODO: get config to verify the pre-caching configuration options
        for aperture, settings in self.configuration['pre-cache'].iteritems():

            logging.info("Pre-caching '{aperture}' aperture with settings: {settings}"
                .format(aperture=aperture, settings=settings))

            # Load the dispersion map for this aperture
            original_dispersion_map = self.dispersion[aperture]

            # Load the new dispersion map that is requested
            if 'dispersion_filename' in settings:

                old_dispersion_filename = settings['dispersion_filename']
                logging.debug("Dispersion filename to interpolate onto is {filename}"
                    .format(filename=old_dispersion_filename))
                cached_dispersion_map = load_model_data(old_dispersion_filename)

                # Over-sample the new dispersion map if necessary
                if 'oversample' in settings and settings['oversample']:
                    
                    logging.debug("Performing oversampling to minimise interpolation losses")

                    num_pixels = len(cached_dispersion_map)
                    cached_dispersion_map = np.linspace(
                        cached_dispersion_map[0],
                        cached_dispersion_map[-1],
                        num_pixels + num_pixels - 1)

                else:
                    logging.debug("No oversampling performed")

            else:
                logging.debug("Using same dispersion map. No oversampling, just smoothing.")

                old_dispersion_filename = self.configuration['models']['dispersion_filenames'][aperture]
                cached_dispersion_map = original_dispersion_map
       
            if old_dispersion_filename.endswith('.cached'):
                # This has already been cached once before. We should warn about this.
                logging.warn("Dispersion filename '{filename}' looks like it may have been cached before."
                    " Continuing to cache to '{cached_filename}' anyways."
                    .format(filename=old_dispersion_filename, cached_filename=cached_dispersion_filename))

            cached_dispersion_filename = self.configuration['models']['dispersion_filenames'][aperture] + '.cached'

            # Go through all the filenames for that aperture
            if isinstance(self.flux_filenames, dict):
                # There is more than one aperture
                flux_filenames = self.flux_filenames[aperture]

            else:
                flux_filenames = self.flux_filenames


            num_flux_filenames = len(flux_filenames)
            for i, flux_filename in enumerate(flux_filenames, 1):

                logging.info("Working on model flux '{flux_filename}' ({i}/{num}).."
                    .format(flux_filename=flux_filename, i=i, num=num_flux_filenames))

                model_flux = load_model_data(flux_filename)

                # For each one: Load the flux, convolve it, and interpolate to the new dispersion
                done_something = False

                # Perform any smoothing
                if 'kernel' in settings:
                    kernel_fwhm = settings['kernel']
                    logging.debug("Convolving with kernel {kernel_fwhm:.3f} Angstroms..".format(kernel_fwhm=kernel_fwhm))

                    # Convert kernel (Angstroms) to pixel size
                    kernel_sigma = kernel_fwhm / (2 * (2*np.log(2))**0.5)
                    
                    # The requested FWHM is in Angstroms, but the dispersion between each
                    # pixel is likely less than an Angstrom, so we must calculate the true
                    # smoothing value
                    
                    true_profile_sigma = kernel_sigma / np.mean(np.diff(original_dispersion_map))
                    model_flux = ndimage.gaussian_filter1d(model_flux, true_profile_sigma)
                    done_something = True

                # Perform any interpolation
                if not np.all(cached_dispersion_map == original_dispersion_map):

                    logging.debug("Interpolating onto new dispersion map..")
                    f = interpolate.interp1d(original_dispersion_map, model_flux, bounds_error=False)

                    done_something = True
                    model_flux = f(cached_dispersion_map)

                # Let's do a sanity check to ensure we've actually *done* something
                if not done_something:
                    logging.warn("Model flux has not been changed by the pre-caching method. Check your configuration file.")

                # Remove non-finite values
                finite_indices = np.isfinite(model_flux)
                model_flux = model_flux[finite_indices]

                # Save the cached dispersion map
                if i == 0:
                    logging.debug("Saving cached dispersion map to {cached_dispersion_filename}".format(cached_dispersion_filename=cached_dispersion_filename))
                    fp = np.memmap(cached_dispersion_filename, dtype=np.float32, mode='w+', shape=cached_dispersion_map[finite_indices].shape)
                    fp[:] = cached_dispersion_map[finite_indices]
                    del fp

                # Save the new flux to file
                cached_flux_filename = flux_filename + '.cached'
                if flux_filename.endswith('.cached'):
                    # Looks like it may have been cached before. We should warn about this.
                    logging.warn("Model flux filename '{filename}' looks like it may have been cached before."
                        " Continuing to cache to '{cached_filename}' anyways."
                        .format(filename=flux_filename, cached_filename=cached_flux_filename))

                # Save cached flux to disk
                logging.debug("Saving cached model flux to {cached_flux_filename}".format(cached_flux_filename=cached_flux_filename))
                fp = np.memmap(cached_flux_filename, dtype=np.float32, mode='w+', shape=model_flux.shape)
                fp[:] = model_flux[:]
                del fp

            # Info.log that shit to recommend altering the yaml file to use cache files instead.
        logging.info("Caching complete. Model filenames for dispersion and flux have been amended to have a"
            " '.cached' extension. You must alter your configuration file to use the cached filenames.")

        return True


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

        return mapped_apertures


def load_model_data(filename, **kwargs):
    """Loads dispersion/flux values from a given filename. This can be either a 1-column ASCII
    file, or a single extension FITS file.

    Inputs
    ------
    filename : `str`
        The filename to load the values from.
    """

    if not os.path.exists(filename):
        raise IOError("filename '{filename}' does not exist".format(filename=filename))

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



        
# coding: utf-8

""" Handles the loading and interpolation of flux models for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["Model"]

# Standard library
import cPickle as pickle
import json
import logging
import multiprocessing
import os
import re
from hashlib import md5
from glob import glob

# Third-party
import numpy as np
import pyfits
import yaml
from scipy import interpolate, ndimage

# Module-specific
from utils import find_spectral_overlap, human_readable_digit 
from specutils import Spectrum1D

logger = logging.getLogger(__name__.split(".")[0])

# This is a hacky (yet necessary) global variable that is used for when pre-loading and
# multiprocessing are employed
_scope_interpolator_ = None

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

    elif filename.endswith('.memmap') \
    or filename.endswith(".cached"):

        # Put in our preferred keyword arguments
        kwargs.setdefault('mode', 'c')
        kwargs.setdefault('dtype', np.float32)
        data = np.memmap(filename, **kwargs)

    else:
        # Assume it must be ASCII.
        data = np.loadtxt(filename, **kwargs)

    return data


def cache_model_point(filename, sigma=None, indices=None):

    # Get the new filename
    basename, extension = os.path.splitext(filename)
    if extension == "cached":
            raise ValueError("previously cached models cannot be re-convolved")

    # Load and smooth the data
    data = load_model_data(filename)
    new_filename = basename + ".cached"

    if sigma is not None:
        data = ndimage.gaussian_filter1d(data, sigma)

    if indices is not None:
        data = data[indices[0]:indices[1]]

    # Save it to disk
    fp = np.memmap(new_filename, dtype=np.float32, mode="w+",
        shape=data.shape)
    fp[:] = data[:]
    del fp

    return new_filename


class Model(object):
    """ A class to represent the data-generating model for spectra """

    def __init__(self, filename):
        """ Initialises a model class from a specified YAML or JSON-style filename """

        if not os.path.exists(filename):
            raise IOError("no model filename '{0}' exists".format(filename))

        parse = json.load if filename.endswith(".json") else yaml.load
        
        # Load the model filename
        with open(filename, "r") as fp:
            self.configuration = parse(fp)

        # Load the dispersions
        self.dispersion = {}
        for aperture in self.apertures:
            self.dispersion[aperture] = load_model_data(self.configuration["model"][aperture]["dispersion_filename"])

        # Model apertures can either have:
        # flux_folder, flux_filename_match
        #   OR
        # points_filename, flux_filename

        # The first is where we will need to fully load the spectra at each point
        # the second is where we need to just load each file
        
        # It can only be one or the other!

        if  "points_filename" in self.configuration["model"][self.apertures[0]] \
        and "flux_filename" in self.configuration["model"][self.apertures[0]]:
            # Check that all apertures are cached and they all refer to the same
            # points filename (if any)

            original_points_filename = self.configuration["model"][self.apertures[0]]["points_filename"]
            for aperture in self.apertures[1:]:
                if self.configuration["model"][aperture].get("points_filename",
                    original_points_filename) != original_points_filename:
                    raise ValueError("points filename must be the same for all cached apertures")

            # Grid points must be pickled data so that the dimension names are known
            with open(original_points_filename, "rb") as fp:
                self.grid_points = pickle.load(fp)

            if len(self.grid_points.dtype.names) == 0:
                raise TypeError("cached grid points filename has no dimension names as columns")

            # Load the fluxes, which must be as memmaps
            global _scope_interpolator_

            num_points = len(self.grid_points)
            fluxes = [] 
            maybe_warn = []
            for i, aperture in enumerate(self.apertures):
                if not "flux_filename" in self.configuration["model"][aperture]:
                    maybe_warn.append(aperture)
                    continue
                fluxes.append(np.memmap(self.configuration["model"][aperture]["flux_filename"], mode="r", dtype=np.float32).reshape((num_points, -1)))

            total_pixels = sum([each.shape[1] for each in fluxes])
            total_expected_pixels = sum(map(len, [self.dispersion[aperture] for aperture in self.apertures]))
            if total_pixels != total_expected_pixels:
                for aperture in maybe_warn:
                    logger.warn("No flux filename specified for {0} aperture".format(aperture))

                raise ValueError("the total flux pixels for a spectrum ({0}) was different to "\
                    "what was expected ({1})".format(total_pixels, total_expected_pixels))

            _scope_interpolator_ = interpolate.LinearNDInterpolator(
                self.grid_points.view(float).reshape((num_points, -1)),
                fluxes[0] if len(fluxes) == 1 else np.hstack(fluxes))

            del fluxes

        elif "flux_folder" in self.configuration["model"][self.apertures[0]] \
        and  "flux_filename_match" in self.configuration["model"][self.apertures[0]]:

            self.grid_points = None
            self.flux_filenames = {}

            dimensions = []
            for i, aperture in enumerate(self.apertures):

                # We will store the filenames and we will load and interpolate on the fly
                folder = self.configuration["model"][aperture]["flux_folder"]
                re_match = self.configuration["model"][aperture]["flux_filename_match"]

                all_filenames = glob(os.path.join(folder, "*"))

                points = []
                matched_filenames = []
                for filename in all_filenames:
                    match = re.match(re_match, os.path.basename(filename))

                    if match is not None:
                        if len(dimensions) == 0:
                            dimensions = sorted(match.groupdict().keys(), key=re_match.index)
           
                        points.append(map(float, match.groups()))
                        matched_filenames.append(filename)

                if self.grid_points is not None and len(points) != len(self.grid_points):
                    raise ValueError("number of model points found in {0} aperture ({1})"
                        " did not match the number in {2} aperture ({3})".format(self.apertures[0],
                        len(self.grid_points), aperture, len(points)))
                       
                # Check the first model flux to ensure it's the same length as the dispersion array
                first_point_flux = load_model_data(matched_filenames[0])
                if len(first_point_flux) != len(self.dispersion[aperture]):
                    raise ValueError("number of dispersion ({0}) and flux ({1}) points in {2} "
                        "aperture do not match".format(len(self.dispersion[aperture]), len(first_point_flux),
                        aperture))

                if i == 0:
                    # Save the grid points as a record array
                    self.grid_points = np.core.records.fromrecords(points, names=dimensions,
                        formats=["f8"]*len(dimensions))
                    self.flux_filenames[aperture] = matched_filenames

                else:
                    sort_indices = np.argsort(map(self.check_grid_point, points))
                    self.flux_filenames[aperture] = [matched_filenames[index] for index in sort_indices]
           
        else:
            raise ValueError("no flux information provided for {0} aperture".format(self.apertures[0]))

        # Pre-compute the grid boundaries
        self.grid_boundaries = {}
        for dimension in self.grid_points.dtype.names:
            self.grid_boundaries[dimension] = np.array([
                np.min(self.grid_points[dimension]),
                np.max(self.grid_points[dimension])
            ])

        # Perform validation checks to ensure there are no forseeable problems
        self.validate()
        return None

    def __repr__(self):
        num_apertures = len(self.apertures)
        num_models = len(self.grid_points) * num_apertures
        num_pixels = sum([len(dispersion) * num_models for dispersion in self.dispersion.values()])
        
        return "{module}.Model({num_models} models; {num_total_parameters} parameters: {num_nuisance_parameters} additional parameters,"\
            " {num_grid_parameters} grid parameters: {parameters}; {num_apertures} apertures: {apertures}; ~{num_pixels} pixels)".format(
            module=self.__module__, num_models=num_models, num_apertures=num_apertures, apertures=', '.join(self.apertures),
            num_pixels=human_readable_digit(num_pixels), num_total_parameters=len(self.dimensions),
            num_nuisance_parameters=len(self.dimensions) - len(self.grid_points.dtype.names), 
            num_grid_parameters=len(self.grid_points.dtype.names), parameters=', '.join(self.grid_points.dtype.names))

    @property
    def hash(self):
        return md5(json.dumps(self.configuration).encode("utf-8")).hexdigest()

    @property
    def apertures(self):
        if hasattr(self, "_mapped_apertures"):
            return self._mapped_apertures
        return self.configuration["model"].keys()

    def validate(self):
        """ Checks aspects of the model for any forseeable errors """

        self._validate_models()
        self._validate_solver()
        self._validate_normalisation()
        self._validate_doppler()
        self._validate_masks()
        return True

    def _check_apertures(self, key):
        for aperture in self.apertures:
            if aperture not in self.configuration[key]:
                raise KeyError("no aperture '{aperture}' listed in {key}, but"
                " it's specified in the models".format(aperture=aperture, key=key))
        return True

    def _validate_normalisation(self):

        self._check_apertures("normalise_observed")

        # Verify the settings for each aperture.
        for aperture in self.apertures:

            settings = self.configuration["normalise_observed"][aperture]

            # Check that settings exist
            if "perform" not in settings:
                raise KeyError("configuration setting 'normalise_observed.{}.perform' not found".format(aperture))

            # If perform is false then we don't need any more details
            if settings["perform"]:

                if "method" not in settings:
                    raise KeyError("configuration setting 'normalise_observed.{}.method' not found".format(aperture))

                method = settings["method"]
                if method == "spline":

                    if "knots" not in settings:
                        raise KeyError("configuration setting 'normalise_observed.{aperture}.knots' not found".format(
                            aperture=aperture))

                    elif not isinstance(settings["knots"], (int, )):
                        # Could be a list-type of rest wavelength points
                        if not isinstance(settings["knots"], (tuple, list, np.ndarray)):
                            raise TypeError("configuration setting 'normalise_observed.{aperture}.knots' is expected"
                                "to be an integer or a list-type of rest wavelength points".format(aperture=aperture))

                        try:
                            map(float, settings["knots"])
                        except (TypeError, ValueError):
                            raise TypeError("configuration setting 'normalise_observed.{aperture}.knots' is expected"
                                "to be an integer or a list-type of rest wavelength points".format(aperture=aperture))

                elif method == "polynomial":

                    if "order" not in settings:
                        raise KeyError("configuration setting 'normalise_observed.{aperture}.order' not found".format(
                            aperture=aperture))

                    elif not isinstance(settings["order"], (float, int)):
                        raise TypeError("configuration setting 'normalise_observed.{aperture}.order'"
                            " is expected to be an integer-like object".format(aperture=aperture))

                else:
                    raise ValueError("configuration setting 'normalise_observed.{aperture}.method' not recognised"
                        " -- must be spline or polynomial".format(aperture=aperture))
        return True


    def _validate_solver(self):
        """ Validates the configuration settings for the solver """

        solver = self.configuration.get("solver", {})
        integers_required = ("sample", "walkers", "burn")
        for key in integers_required:
            if key not in solver:
                raise KeyError("configuration setting 'solver.{}' not found".format(key))

            try:
                int(solver[key])
            except (ValueError, TypeError):
                raise TypeError("configuration setting 'solver.{}' must be an integer-like type".format(key))

        if solver.get("optimise", True):
            # Check for initial_samples
            if "initial_samples" not in solver:
                raise KeyError("configuration setting 'solver.initial_samples' not found")
            try:
                int(solver["initial_samples"])
            except (ValueError, TypeError):
                raise TypeError("configuration setting 'solver.initial_samples' must be an integer-like type")

        if "threads" in solver and not isinstance(solver["threads"], (float, int)):
            raise TypeError("configuration setting 'solver.threads' must be an integer-like type")
        return True


    def _validate_doppler(self):
        """ Validates the configuration settings for doppler shifts """

        self._check_apertures("doppler_shift")
        for aperture in self.apertures:
            if 'perform' not in self.configuration['doppler_shift'][aperture]:
                raise KeyError("configuration setting 'doppler_shift.{}.perform' not found".format(aperture))
        return True


    def _validate_smoothing(self):

        self._check_apertures("smooth_model_flux")

        for aperture in self.apertures:
            settings = self.configuration["smooth_model_flux"][aperture]
            if "perform" not in settings:
                raise KeyError("configuration setting 'smooth_model_flux.{0}.perform' not found".format(
                    aperture))

            for setting, value in settings.iteritems():
                if setting == "perform" and not value:
                    break
                if setting == "kernel" and not isinstance(value, (int, float)):
                    raise TypeError("configuration setting 'smooth_model_flux.{0}.kernel' is"
                        " expected to be a float-type".format(aperture))
        return True


    def _validate_models(self):
        """ Validates the configuration settings for the model """

        if "model" not in self.configuration.keys():
            raise KeyError("no 'model' attribute found in model file")

        for aperture in self.apertures:
            if "." in aperture:
                raise ValueError("aperture name '{0}' cannot contain a full-stop character".format(aperture))


    def _validate_masks(self):
        """ Validate the provided mask settings """

        # Masks are optional
        if "masks" not in self.configuration.keys():
            return True

        for aperture in self.configuration["masks"].keys():
            if aperture not in self.apertures: continue

            if not isinstance(self.configuration["masks"][aperture], (tuple, list)):
                raise TypeError("masks must be a list of regions (e.g., [start, stop])")

            for region in self.configuration["masks"][aperture]:
                if not isinstance(region, (list, tuple)) or len(region) != 2:
                    raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")

                try:
                    map(float, region)
                except TypeError:
                    raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")

        return True


    @property
    def dimensions(self):
        """ Returns the dimension names for a given model """

        if hasattr(self, "_dimensions"):
            return self._dimensions

        dimensions = [] + list(self.grid_points.dtype.names)
        for dimension in self.configuration.keys():

            # Protected keywords
            if dimension in ("solver", "model"): continue

            # Add all smoothing-related dimensions
            elif dimension == "smooth_model_flux":
                for aperture in self.apertures:
                    if self.configuration[dimension][aperture]["perform"] \
                    and self.configuration[dimension][aperture]["kernel"] == "free":
                        dimensions.append("smooth_model_flux.{0}.kernel".format(aperture))

            # Add doppler-related dimensions
            elif dimension == "doppler_shift":
                for aperture in self.apertures:
                    if self.configuration[dimension][aperture]["perform"]:
                        dimensions.append("doppler_shift.{0}".format(aperture))

            # Add normalisation-related dimensions
            elif dimension == "normalise_observed":
                for aperture in self.apertures:
                    if self.configuration[dimension][aperture]["perform"]:
                        if self.configuration[dimension][aperture].get("method", "polynomial") == "polynomial":
                            dimensions.extend(
                                ["normalise_observed.{0}.a{1}".format(aperture, i) \
                                    for i in xrange(self.configuration["normalise_observed"][aperture]["order"] + 1)])

                        #else: #spline
                        #    dimensions.append("normalise_observed.{0}.s".format(aperture))

        # Append jitter dimensions
        for aperture in self.apertures:
            dimensions.append("jitter.{0}".format(aperture))
        
        # Cache for future
        setattr(self, "_dimensions", dimensions)
        
        return dimensions


    def cache(self, grid_points_filename, dispersion_filenames, flux_filename, wavelengths=None, smoothing_kernels=None,
        sampling=None):
        """ Cache the model dispersions and fluxes for faster read access at runtime.
        Spectra can be pre-convolved

        """

        if not isinstance(smoothing_kernels, dict):
            raise TypeError("smoothing kernels must be a dictionary-type with apertures as keys"\
                " and kernels as values")

        if not isinstance(wavelengths, dict):
            raise TypeError("wavelengths must be a dictionary-type with apertures as keys")

        if sampling is None:
            sampling = dict(zip(self.apertures, np.ones(len(self.apertures))))

        # We need to know how big our arrays will be
        n_points = len(self.grid_points)

        if wavelengths is None:
            n_pixels = []
            wavelength_indices = {}
            for aperture in self.apertures:
                pixels = len(self.dispersion[aperture])
                n_pixels.append(int(np.ceil(pixels/float(sampling[aperture]))))
                wavelength_indices[aperture] = np.array([0, pixels])

        else:
            n_pixels = []
            wavelength_indices = {}
            for aperture in self.apertures:
                start, end = self.dispersion[aperture].searchsorted(wavelengths[aperture])
                wavelength_indices[aperture] = np.array([start, end])

                pixels = end - start 
                n_pixels.append(int(np.ceil(pixels/float(sampling[aperture]))))

        # Create the grid points filename
        with open(grid_points_filename, "wb") as fp:
            pickle.dump(self.grid_points, fp)

        # Create empty memmap
        flux = np.memmap(flux_filename, dtype=np.float32, mode="w+", shape=(n_points, sum(n_pixels)))
        for i in xrange(n_points):

            logger.info("Caching point {0} of {1} ({2:.1f}%)".format(i+1, n_points, 100*(i+1.)/n_points))
            for j, aperture in enumerate(self.apertures):

                sj, ej = map(int, map(sum, [n_pixels[:j], n_pixels[:j+1]]))

                # Get the flux
                si, ei = wavelength_indices[aperture]
                aperture_flux = load_model_data(self.flux_filenames[aperture][i])

                # Do we need to convolve it first?
                if smoothing_kernels is not None and smoothing_kernels.has_key(aperture):
                    sigma = (smoothing_kernels[aperture]/(2 * (2*np.log(2))**0.5))/np.mean(np.diff(self.dispersion[aperture]))
                    aperture_flux = ndimage.gaussian_filter1d(aperture_flux, sigma)

                # Do we need to resample?
                flux[i, sj:ej] = aperture_flux[si:ei:sampling[aperture]]

        # Save the resampled dispersion arrays to disk
        for aperture, dispersion_filename in dispersion_filenames.iteritems():
            si, ei = wavelength_indices[aperture]

            disp = np.memmap(dispersion_filename, dtype=np.float32, mode="w+",
                shape=self.dispersion[aperture][si:ei:sampling[aperture]].shape)
            disp[:] = self.dispersion[aperture][si:ei:sampling[aperture]]
            del disp

        # Now we need to save the flux to disk
        del flux

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

        if len(point) != len(self.grid_points.dtype.names):
            raise ValueError("point length ({0}) is incompatible with grid shape ({1})"
                .format(length=len(point), shape=len(self.grid_points.dtype.names)))

        indices = set(np.arange(len(self.grid_points)))
        for i, point_value in enumerate(point):
            difference = np.unique(self.grid_points[:, i] - point_value)

            if sum(difference > 0) * sum(difference < 0) == 0:
                return ValueError("point ({0}) outside of the grid boundaries".format(point_value))

            limit_min = difference[np.where(difference < 0)][-n:][0] + point_value
            limit_max = difference[np.where(difference > 0)][:n][-1] + point_value
    
            these_indices = np.where((limit_max >= self.grid_points[:, i]) * (self.grid_points[:, i] >= limit_min))[0]
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

        num_dimensions = len(self.grid_points.dtype.names)
        index = np.all(self.grid_points.view(np.float).reshape((-1, num_dimensions)) == np.array([point]).view(np.float),
            axis=-1)
        
        if not any(index):
            return False
        return np.where(index)[0][0]


    def interpolate_flux(self, point, **kwargs):
        """Interpolates through the grid of models to the given `point` and returns
        the interpolated flux.

        Inputs
        ------
        point : list of `float` values
            The point to interpolate to.
        """

        global _scope_interpolator_
        if _scope_interpolator_ is not None:

            interpolated_flux = _scope_interpolator_(*point)
            
            interpolated_fluxes = {}
            num_pixels = map(len, [self.dispersion[aperture] for aperture in self.apertures])
            for i, aperture in enumerate(self.apertures):
                si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
                interpolated_fluxes[aperture] = interpolated_flux[si:ei]
            return interpolated_fluxes

        interpolated_fluxes = {}
        indices = self.get_nearest_neighbours(point)
        for aperture in self.apertures:
            
            flux = np.zeros((len(indices), len(self.dispersion[aperture])))
            flux[:] = np.nan

            # Load the flux points
            for i, index in enumerate(indices):
                aperture_flux[i, :] = load_model_data(self.flux_filenames[aperture][index])
            
            try:
                interpolated_flux[aperture] = interpolate.griddata(self.grid_points[indices],
                    aperture_flux, [point], **kwargs).flatten()

            except:
                raise ValueError("could not interpolate flux point -- it is likely outside the grid boundaries")
    

    def map_apertures(self, observations):
        """References model spectra to each observed spectra based on their wavelengths.

        Inputs
        ------
        observations : a list of observed spectra to map
        """

        mapped_apertures = []
        for i, spectrum in enumerate(observations):

            # Initialise the list
            apertures_found = []

            observed_wlmin = np.min(spectrum.disp)
            observed_wlmax = np.max(spectrum.disp)

            for model_aperture, model_dispersion in self.dispersion.iteritems():

                model_wlmin = np.min(model_dispersion)
                model_wlmax = np.max(model_dispersion)

                # Is there overlap?
                if (model_wlmin <= observed_wlmin and observed_wlmax <= model_wlmax) \
                or (observed_wlmin <= model_wlmin and model_wlmax <= observed_wlmax) \
                or (model_wlmin <= observed_wlmin and (observed_wlmin <= model_wlmax and model_wlmax <= observed_wlmax)) \
                or ((observed_wlmin <= model_wlmin and model_wlmin <= observed_wlmax) and observed_wlmax <= model_wlmax):
                    apertures_found.append(model_aperture)

            if len(apertures_found) == 0:
                raise ValueError("no model apertures found for observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                    .format(wl_start=observed_wlmin, wl_end=observed_wlmax))

            elif len(apertures_found) > 1:
                index = np.argmin(np.abs(np.mean(spectrum.disp) - map(np.mean, [self.dispersion[aperture] for aperture in apertures_found])))
                apertures_found = [apertures_found[index]]

                logging.warn("Multiple model apertures found for observed aperture {0} ({1:.0f} to {2:.0f}). Using '{0}'"
                    " because it's closest by mean dispersion.".format(i, observed_wlmin, observed_wlmax, apertures_found[0]))


            mapped_apertures.append(apertures_found[0])

        # Check that the mean pixel size in the model dispersion maps is smaller than the observed dispersion maps
        for aperture, spectrum in zip(mapped_apertures, observations):

            mean_observed_pixel_size = np.mean(np.diff(spectrum.disp))
            mean_model_pixel_size = np.mean(np.diff(self.dispersion[aperture]))
            if mean_model_pixel_size > mean_observed_pixel_size:
                raise ValueError("the mean model pixel size in the {aperture} aperture is larger than the mean"
                    " pixel size in the observed dispersion map from {wl_start:.1f} to {wl_end:.1f}"
                    .format(aperture=aperture, wl_start=np.min(spectrum.disp), wl_end=np.max(spectrum.disp)))

        # Keep an internal reference of the aperture mapping
        self._mapped_apertures = mapped_apertures

        # Should we update our internal .apertures and .dimensions?
        return mapped_apertures


    def __call__(self, observations=None, **kwargs):
        """ Generates model fluxes for a provided set of parameters """

        # Get the grid point and interpolate
        point = [kwargs[parameter] for parameter in self.grid_points.dtype.names]
        interpolated_flux = self.interpolate_flux(point)

        all_non_finite = lambda fluxes: np.all(~np.isfinite(fluxes))
        if any(map(all_non_finite, interpolated_flux.values())):
            raise ValueError("interpolated aperture contained all non-finite flux vlues")

        # Transform the spectra
        model_spectra = {}
        for aperture, interpolated_flux in interpolated_flux.iteritems():
                
            model_spectra[aperture] = Spectrum1D(disp=self.dispersion[aperture].copy(), flux=interpolated_flux.copy())

            # Any synthetic smoothing to apply?
            smoothing_kwarg = "smooth_model_flux.{0}.kernel".format(aperture)

            # Is the smoothing a free parameter?
            fwhm = 0
            if smoothing_kwarg in kwargs:
                fwhm = kwargs[smoothing_kwarg]

            elif self.configuration["smooth_model_flux"][aperture]["perform"]:
                # Apply a single smoothing value
                fwhm = self.configuration["smooth_model_flux"][aperture]["kernel"]
            
            if fwhm > 0:
                profile_sigma = fwhm / (2.*(2*np.log(2))**0.5)
                true_profile_sigma = profile_sigma / np.mean(np.diff(model_spectra[aperture].disp))
                model_spectra[aperture].flux = ndimage.gaussian_filter1d(model_spectra[aperture].flux, true_profile_sigma)

            # Doppler shift the spectra
            doppler_shift_kwarg = "doppler_shift.{0}".format(aperture)
            if doppler_shift_kwarg in kwargs:
                c, v = 299792458e-3, kwargs[doppler_shift_kwarg]
                model_spectra[aperture].disp *= np.sqrt((1 + v/c)/(1 - v/c))

            # Interpolate synthetic to observed dispersion map
            if observations is not None:
                model_spectra[aperture] = model_spectra[aperture].interpolate(
                    observations[self.apertures.index(aperture)].disp)

            # Scale to the data
            if self.configuration["normalise_observed"][aperture]["perform"]:
            
                if self.configuration["normalise_observed"][aperture].get("method", "polynomial") == "polynomial":

                    # Since we need to perform normalisation, the normalisation coefficients should be in the kwargs
                    num_coefficients_expected = self.configuration["normalise_observed"][aperture]["order"] + 1
                    coefficients = [kwargs["normalise_observed.{aperture}.a{n}".format(aperture=aperture, n=n)] \
                        for n in xrange(num_coefficients_expected)]
                    model_spectra[aperture].flux *= np.polyval(coefficients, model_spectra[aperture].disp)
    
                else: #spline
                    if observations is not None:
                        num_knots = self.configuration["normalise_observed"][aperture]["knots"]
                        observed_aperture = observations[self.apertures.index(aperture)]
                        
                        # Divide the observed spectrum by the model aperture spectrum
                        continuum = observed_aperture.flux/model_spectra[aperture].flux

                        # Apply a mask
                        mask = np.where(self.masks({aperture: model_spectra[aperture]}, **kwargs)[aperture] == 0)[0]
                        continuum[mask] = np.nan

                        # Fit a spline function to the *finite* continuum points, since the model spectra is interpolated
                        # to all observed pixels (regardless of how much overlap there is)
                        finite = np.isfinite(continuum)

                        if isinstance(num_knots, (float, int)):
                            # Produce equi-spaced internal knot points
                            # Divide the spectral range by <N> + 1
                            spacing = np.ptp(observed_aperture.disp[finite])/(num_knots + 1.)
                            knots = np.arange(observed_aperture.disp[finite][0] + spacing,
                                observed_aperture.disp[finite][-1], spacing)[:num_knots]

                        else:
                            # If num_knots is not actually a number then it is a list-type of actual knot points that
                            # have been specified by the user
                            knots = num_knots

                        # TODO: Should S be free?
                        tck = interpolate.splrep(observed_aperture.disp[finite], continuum[finite],
                            w=1./observed_aperture.uncertainty[finite], t=knots)
                        #s=kwargs["normalise_observed.{0}.s".format(aperture)])

                        # Scale the model by the continuum function
                        model_spectra[aperture].flux[finite] *= interpolate.splev(observed_aperture.disp[finite], tck)

        return [model_spectra[aperture] for aperture in self.apertures]


    def masks(self, model_spectra, **kwargs):
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
                    mask = np.ones(len(spectrum.disp))
                    if self.configuration["masks"][aperture] is not None:
                        for region in self.configuration["masks"][aperture]:

                            region = np.array(region)

                            if "doppler_shift.{0}".format(aperture) in kwargs:
                                c, v = 299792458e-3, kwargs["doppler_shift.{0}".format(aperture)]
                                region *= np.sqrt((1 + v/c)/(1 - v/c))
                                
                            index_start, index_end = np.searchsorted(spectrum.disp, region)
                            mask[index_start:index_end] = 0

                    masks[aperture] = mask

        return masks


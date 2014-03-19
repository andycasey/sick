# coding: utf-8

""" Handles the loading and interpolation of flux models for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

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

__all__ = ["Model"]

# This is a hacky global variable that is used for when pre-loading and
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

        # Wrapping the data as a new array so that it is pickleable across
        # multiprocessing without the need to share the memmap. See:
        # http://stackoverflow.com/questions/15986133/why-does-a-memmap-need-a-filename-when-i-try-to-reshape-a-numpy-array
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
    fp = np.memmap(new_filename, dtype="float32", mode="w+",
        shape=data.shape)
    fp[:] = data[:]
    del fp

    return new_filename


def mapper(*x):
    return apply(x[0], x[1:])


class Model(object):
    """ Model class for SCOPE """

    def __init__(self, filename):

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
            _scope_interpolator_ = {}

            num_points = len(self.grid_points)
            for aperture in self.apertures:

                flux_filename = self.configuration["model"][aperture]["flux_filename"]
                aperture_flux = np.memmap(flux_filename, mode="r", dtype=np.float32).reshape((num_points, -1))
                _scope_interpolator_[aperture] = interpolate.LinearNDInterpolator(
                    self.grid_points.view(float).reshape((num_points, -1)),
                    aperture_flux)

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
                            
                            groups = match.groups()
                            groupdict = match.groupdict()

                            for value in match.groupdict().itervalues():
                                if groups.count(value) > 1: break
                                dimensions.append(match.groupdict().keys()[groups.index(value)])

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
        
        return "{module}.Model({num_models} models, {num_apertures} apertures: {apertures}, {num_parameters}"\
            " parameters: {parameters}, ~{num_pixels} pixels)".format(module=self.__module__, num_models=num_models,
            num_apertures=num_apertures, apertures=', '.join(self.apertures), num_pixels=human_readable_digit(num_pixels),
            num_parameters=len(self.grid_points.dtype.names), parameters=', '.join(self.grid_points.dtype.names))

    @property
    def hash(self):
        return md5(json.dumps(self.configuration).encode("utf-8")).hexdigest()

    @property
    def apertures(self):
        return self.configuration["model"].keys()

    def validate(self):
        """ Checks aspects of the model for any forseeable errors """

        self._validate_models()
        self._validate_solver()
        self._validate_normalisation()
        self._validate_solver()
        self._validate_doppler()
        self._validate_masks()
        self._validate_weights()
    
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
                raise KeyError("configuration setting 'normalise_observed.{aperture}.perform' not found".format(aperture=aperture))

            # If perform is false then we don't need order
            if settings["perform"]:
                if "order" not in settings:
                    raise KeyError("configuration setting 'normalise_observed.{aperture}.{required_setting}' not found".format(
                        aperture=aperture, required_setting=required_setting))

                elif not isinstance(settings["order"], (float, int)):
                    raise TypeError("configuration setting 'normalise_observed.{aperture}.order'"
                        " is expected to be an integer-like object".format(aperture=aperture))

        return True


    def _validate_solver(self):

        if "solver" not in self.configuration.keys():
            raise KeyError("no solver information provided in configuration")

        solver = self.configuration["solver"]
        available_methods = ("fmin_powell", "emcee")

        if solver["method"] not in available_methods:
            raise ValueError("solver method '{0}' is unsupported. Available methods are: {1}".format(
                solver["method"], ", ".join(available_methods)))

        if "threads" in solver and not isinstance(solver["threads"], (float, int)):
            raise TypeError("solver.threads must be an integer-like type")

        return True


    def _validate_doppler(self):

        self._check_apertures("doppler_shift")

        priors_to_expect = []
        for aperture in self.apertures:
            if 'perform' not in self.configuration['doppler_shift'][aperture]:
                raise KeyError("configuration setting 'doppler_shift.{aperture}.perform' not found".format(aperture=aperture))

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

        if "model" not in self.configuration.keys():
            raise KeyError("no 'model' attribute found in model file")

        for aperture in self.apertures:
            if "." in aperture:
                raise ValueError("aperture name '{0}' cannot contain a full-stop character".format(aperture))

        # Check dispersion maps of standards don't overlap
        overlap_wavelength = find_spectral_overlap(self.dispersion.values())
        if overlap_wavelength is not None:
            raise ValueError("dispersion maps overlap near {0:.0f}".format(overlap_wavelength))


    def _validate_masks(self):

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


    def _validate_weights(self):

        if "weights" not in self.configuration.keys():
            return True

        for aperture in self.configuration["weights"]:
            if aperture not in self.apertures: continue

            test_weighting_func = lambda disp, flux: eval(self.configuration["weights"][aperture],
                {"disp": disp, "flux": flux, "np": np})

            try:
                test_weighting_func(1, 1)
                test_weighting_func(np.arange(10), np.ones(10))

            except:
                raise ValueError("weighting function for {0} aperture is improper".format(aperture))

        return True


    @property
    def dimensions(self):
        """Returns all dimension names for a given configuration, which can
        include both implicit and explicit priors."""

        if hasattr(self, "_dimensions"):
            return self._dimensions

        # Get explicit priors
        dimensions = [] + self.grid_points.dtype.names
        for dimension in self.configuration["priors"].keys():
            if dimension.startswith("doppler_shift") \
            or dimension.startswith("smooth_model_flux"):
                aperture = dimension.split(".")[1]
                if aperture not in self.apertures:
                    continue
            if dimension in dimensions: continue

            dimensions.append(dimension)

        # Get implicit normalisation priors
        for aperture in self.apertures:     
            if self.configuration["normalise_observed"][aperture]["perform"]:
                dimensions.extend(
                    ["normalise_observed.{0}.a{1}".format(aperture, i) \
                        for i in xrange(self.configuration["normalise_observed"][aperture]["order"] + 1)])

        # Append jitter dimension
        if self.configuration["solver"].get("walkers", 1) > 1:
            for aperture in self.apertures:
                dimensions.append("jitter.{0}".format(aperture))
        
        setattr(self, "_dimensions", dimensions)
        
        return dimensions


    def cache_models(self, smoothing_kernels=None, wavelength_slices=None, sample_rate=1,
        threads=None, **kwargs):
        """ Cache the model dispersions and fluxes for faster read access at runtime.
        Spectra can be pre-convolved

        """

        threads = 1 if threads is None else threads
        profile_sigma = lambda k, d: (k / (2 * (2*np.log(2))**0.5))/np.mean(np.diff(d))

        pool = multiprocessing.Pool(threads)

        if isinstance(self.flux_filenames, dict):
            for aperture in self.flux_filenames.keys():
                if aperture not in kernel.keys():
                    pool.close()
                    raise ValueError("no kernel provided for '{0}' aperture".format(aperture))

            # For each aperture, load all the spectra, convolve it, get a new filename, write cached file to disk
            for beam, flux_filenames in self.flux_filenames.iteritems():
                
                sigma = profile_sigma(kernel[beam], self.dispersion[beam])
                
                if beam in self.configuration["masks"]:
                    min_wl, max_wl = np.min(self.configuration["masks"][beam]), np.max(self.configuration["masks"][beam])
                    indices = self.dispersion[beam].searchsorted([min_wl, max_wl])

                else:
                    indices = None

                # Cache the dispersion filename
                pool.apply_async(mapper, args=(cache_model_point, self.configuration["model"]["dispersion_filenames"][beam], None, indices))

                for flux_filename in flux_filenames:
                    pool.apply_async(mapper, args=(cache_model_point, flux_filename, sigma, indices))

        else:
            raise NotImplementedError

            sigma = profile_sigma(kernel[beam], self.dispersion)
            for flux_filename in self.flux_filenames:
                pool.apply_async(mapper, args=(cache_model_point, flux_filename, sigma))

        pool.close()
        pool.join()

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

        index = np.all(self.grid_points == point, axis=-1)
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

            interpolated_fluxes = {}
            for i, aperture in enumerate(self.apertures):
                interpolated_fluxes[aperture] = _scope_interpolator_(point)
            return interpolated_fluxes

        indices = self.get_nearest_neighbours(point)

        interpolated_fluxes = {}
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

        return interpolated_fluxes
    

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

        # Should we update our internal .apertures and .dimensions?
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
        grid_point = [kwargs[stellar_parameter] for stellar_parameter in self.grid_points.dtype.names]

        # Interpolate the flux
        try:
            interpolated_flux = self.interpolate_flux(grid_point)

        except:
            logging.debug("No model flux could be determined for {0}".format(
                ", ".join(["{0} = {1:.2f}".format(parameter, value) \
                    for parameter, value in zip(self.grid_points.dtype.names, grid_point)])))
            return None

        logging.debug("Interpolated model flux at {0}".format(
            ", ".join(["{0} = {1:.2f}".format(parameter, value) \
                for parameter, value in zip(self.grid_points.dtype.names, grid_point)])))

        if interpolated_flux == {}:
            logging.debug("No apertures of interpolated flux found")
            return None

        for aperture, flux in interpolated_flux.iteritems():
            if np.all(~np.isfinite(flux)):
                logging.debug("{0} aperture returned all non-finite values for interpolated flux".format(aperture))
                return None

        # Create spectra
        model_spectra = {}
        for aperture, interpolated_flux in interpolated_flux.iteritems():
            model_spectra[aperture] = Spectrum1D(disp=self.dispersion[aperture],
                flux=interpolated_flux)

        # Any synthetic smoothing to apply?
        for aperture in self._mapped_apertures:
            key = "smooth_model_flux.{aperture}.kernel".format(aperture=aperture)

            # Is the smoothing a free parameter?
            if key in kwargs:
                # Ensure valid smoothing value
                if kwargs[key] < 0: return None
                model_spectra[aperture] = model_spectra[aperture].gaussian_smooth(kwargs[key])

            elif self.configuration["smooth_model_flux"][aperture]["perform"]:
                # Apply a single smoothing value
                kernel = self.configuration["smooth_model_flux"][aperture]["kernel"]
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

                if np.all(modified_spectrum.flux < 0):
                    return None

            # Any doppler shift to perform?
            if self.configuration["doppler_shift"][aperture]["perform"]:

                velocity = kwargs["doppler_shift.{0}".format(aperture)]
                modified_spectrum = modified_spectrum.doppler_shift(-velocity)

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


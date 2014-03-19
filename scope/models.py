# coding: utf-8

""" Handles the loading and interpolation of flux models for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Standard library
import logging
import os
import re
from glob import glob
import multiprocessing

# Third-party
import numpy as np
import pyfits
from scipy import interpolate, ndimage, spatial
import yaml

# Module
from utils import human_readable_digit
from specutils import Spectrum1D


__all__ = ['Model', 'load_model_data', "_scope_interpolator_"]

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

    elif filename.endswith('.cached'):

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

        # Dispersions
        self.dispersion = {}
        for beam, dispersion_filename in self.configuration['models']['dispersion_filenames'].iteritems():
            self.dispersion[beam] = load_model_data(dispersion_filename)

        grid_points = {}
        flux_filenames = {}

        # Read the points from dispersion filenames
        for beam in self.configuration['models']['dispersion_filenames']:
            folder = self.configuration['models']['flux_filenames'][beam]['folder']
            re_match = self.configuration['models']['flux_filenames'][beam]['re_match']

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

            # Check the first model flux to ensure it's the same length as the dispersion array
            first_point_flux = load_model_data(matched_filenames[0])
            if len(first_point_flux) != len(self.dispersion[beam]):
                raise ValueError("number of dispersion ({0}) and flux ({1}) points in {2} aperture do not match".format(
                    len(self.dispersion[beam]), len(first_point_flux), beam))

            grid_points[beam] = points
            flux_filenames[beam] = matched_filenames

        first_beam = self.configuration['models']['dispersion_filenames'].keys()[0]
        self.grid_points = np.array(grid_points[first_beam])
        self._tree = spatial.cKDTree(self.grid_points)

        self.grid_boundaries = {}
        for i, colname in enumerate(self.colnames):
            self.grid_boundaries[colname] = np.array([
                np.min(self.grid_points[:, i]),
                np.max(self.grid_points[:, i])
                ])

        # If it's just the one beam, it's easy!        
        if len(self.configuration['models']['dispersion_filenames'].keys()) == 1:
            self.flux_filenames = flux_filenames[first_beam]
            return None
        
        else:
            self.flux_filenames = {first_beam: flux_filenames[first_beam]}

        # Put all points and filenames on the one scale
        for beam in self.configuration['models']['dispersion_filenames'].keys()[1:]:
            
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
        
        return '{module}.Model({num_models} models, {num_apertures} apertures: "{apertures}", {num_parameters} parameters: "{parameters}", ~{num_pixels} pixels)'.format(
            module=self.__module__, num_models=num_models, num_apertures=num_apertures, apertures=', '.join(self.dispersion.keys()),
            num_pixels=human_readable_digit(num_pixels), num_parameters=self.grid_points.shape[1],
            parameters=', '.join(self.colnames))


    @property
    def apertures(self):
        return self.configuration['models']['dispersion_filenames'].keys()

    def validate(self):

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

        self._check_apertures("normalisation")

        # Verify the settings for each aperture.
        for aperture in self.apertures:

            settings = configuration['normalise_observed'][aperture]

            # Check that settings exist
            if 'perform' not in settings:
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

        if "models" not in self.configuration.keys():
            raise KeyError("no `models` attribute found in model file")

        settings = self.configuration["models"]

        for aperture in self.apertures:
            if "." in aperture:
                raise ValueError("aperture name '{0}' cannot contain a full-stop character".format(aperture))

        required_keys = ("dispersion_filenames", "flux_filenames")
        for key in required_keys:
            if key not in settings.keys():
                raise KeyError("required 'models.{0}' attribute not found in model".format(key))

        # Check for missing keys
        missing_keys = [x for x in settings["dispersion_filenames"].keys() if x not in settings["flux_filenames"]]
        if len(missing_keys) > 0:
            raise KeyError("missing flux filenames for {0} aperture(s)".format(", ".join(missing_keys)))

        # Check dispersion maps of standards don't overlap
        overlap_wavelength = utils.find_spectral_overlap(self.dispersions.values())
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


    def check_aperture_names(configuration, key):
        """Checks that all the apertures specified in the models
        exist in the given sub-config `key`.

        Inputs
        ------
        `configuration` : dict
            A dictionary configuration for SCOPE.

        `key` : str
            The sub-key to check (e.g. 'normalise_observed')
        """

        for aperture in self.apertures:
            if aperture not in configuration[key]:
                raise KeyError("no aperture '{aperture}' listed in {key}, but"
                " it's specified in the models".format(aperture=aperture, key=key))
        return True


    @property
    def dimensions(self):
        """Returns all dimension names for a given configuration, which can
        include both implicit and explicit priors."""

        if hasattr(self, "_dimensions"):
            return self._dimensions

        # Get the actual apertures we're going to use
        useful_apertures = self.configuration["models"]["dispersion_filenames"].keys()

        # Get explicit priors
        dimensions = [] + self.colnames
        for dimension in self.configuration["priors"].keys():
            if dimension.startswith("doppler_shift") \
            or dimension.startswith("smooth_model_flux"):
                aperture = dimension.split(".")[1]
                if aperture not in useful_apertures:
                    continue
            if dimension in dimensions: continue

            dimensions.append(dimension)

        # Get implicit normalisation priors
        for aperture in useful_apertures:     
            if self.configuration["normalise_observed"][aperture]["perform"]:
                dimensions.extend(
                    ["normalise_observed.{0}.a{1}".format(aperture, i) \
                        for i in xrange(self.configuration["normalise_observed"][aperture]["order"] + 1)])

        # Append jitter dimension
        if self.configuration["solver"].get("walkers", 1) > 1:
            for aperture in useful_apertures:
                dimensions.append("jitter.{0}".format(aperture))
        
        setattr(self, "_dimensions", dimensions)
        
        return dimensions


    def pre_load(self):
        """ Pre loads all spectra to an interpolator for future speedyness. """

        if isinstance(self.flux_filenames, dict):

            n_flux_points = np.zeros(len(self.dispersion_filenames.keys()))
            for i, beam in enumerate(self.dispersion_filenames.keys()):
                if i == 0:
                    n_models = len(self.flux_filenames[beam])
                n_flux_points[i] = len(load_model_data(self.flux_filenames[beam][0]))

            flux = np.empty((n_models, sum(n_flux_points)))
            for i, beam in enumerate(self.dispersion_filenames.keys()):

                si, ei = map(sum, [n_flux_points[:i], n_flux_points[:i+1]])
                flux[i, si:ei] = load_model_data(self.flux_filenames[beam][i], mode="r")

            # We will need this
            self._beam_flux_points = n_flux_points

        else:
            
            n_models = len(self.flux_filenames)
            n_flux_points = len(load_model_data(self.flux_filenames[0]))

            # Load the flux
            flux = np.empty((n_models, n_flux_points))
            for i, filename in enumerate(self.flux_filenames):
                flux[i, :] = load_model_data(filename, mode="r")

            self._beam_flux_points = np.array([n_flux_points])

        # Create the interpolator
        global _scope_interpolator_
        _scope_interpolator_ = interpolate.LinearNDInterpolator(self.grid_points, flux)

        self._pre_loaded = True
        return True


    def cache(self, kernel, threads=None, **kwargs):
        """ Pre-convolve the model fluxes and save their convolved flux to memory-mapped
        files for faster read access. 

        """

        threads = 1 if threads is None else threads
        profile_sigma = lambda k, d: (k / (2 * (2*np.log(2))**0.5))/np.mean(np.diff(d))

        pool = multiprocessing.Pool(threads)

        if isinstance(self.flux_filenames, dict):
            for aperture in self.flux_filenames.keys():
                if aperture not in kernel.keys():
                    pool.close()
                    raise ValueError("no kernel provided for '{0}' aperture".format(aperture))

            # For each beam, load all the spectra, convolve it, get a new filename, write cached file to disk
            for beam, flux_filenames in self.flux_filenames.iteritems():
                
                sigma = profile_sigma(kernel[beam], self.dispersion[beam])
                
                if beam in self.configuration["masks"]:
                    min_wl, max_wl = np.min(self.configuration["masks"][beam]), np.max(self.configuration["masks"][beam])
                    indices = self.dispersion[beam].searchsorted([min_wl, max_wl])

                else:
                    indices = None

                # Cache the dispersion filename
                pool.apply_async(mapper, args=(cache_model_point, self.configuration["models"]["dispersion_filenames"][beam], None, indices))

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

        if len(point) != self.grid_points.shape[1]:
            raise ValueError("point length ({length}) is incompatible with grid shape ({shape})"
                .format(length=len(point), shape=self.grid_points.shape))

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


    def interpolate_flux(self, point, beams='all', kind='linear', max_neighbours=3, **kwargs):
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


        if beams is 'all':
            beams = self.dispersion.keys()

        elif not isinstance(beams, (list, tuple)):
            beams = [beams]

        for beam in beams:
            if beam not in self.dispersion.keys():
                raise ValueError("could not find '{beam}' beam".format(beam=beam))

        if hasattr(self, "_pre_loaded"):

            
            interpolated_flux = _scope_interpolator_(*point)
            interpolated_beams = {}
            for i, beam in enumerate(beams):
                si, ei = map(sum, [self._beam_flux_points[:i], self._beam_flux_points[:i+1]])
                interpolated_beams[beam] = interpolated_flux[si:ei]

            return interpolated_beams

        for n in xrange(1, max_neighbours):

            indices = self.get_nearest_neighbours(point, n=n)
            if len(indices) == len(self.grid_points):
                raise ValueError("could not interpolate flux point -- hard limit hit")

            failed = False
            interpolated_flux = {}
            for beam in beams:
                beam_flux = np.zeros((len(indices), len(self.dispersion[beam])))
                beam_flux[:] = np.nan

                # Load the flux points
                flux_data = self.flux_filenames if len(self.dispersion.keys()) == 1 else self.flux_filenames[beam]
                for i, index in enumerate(indices):
                    beam_flux[i, :] = load_model_data(flux_data[index])
                
                try:
                    interpolated_flux[beam] = interpolate.griddata(
                        self.grid_points[indices], beam_flux, [point], **kwargs).flatten()

                except:
                    
                    failed = True
                    if n == max_neighbours:
                        raise ValueError("could not interpolate flux point -- it is likely outside the grid boundaries")

                    break

            if failed:
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

        if interpolated_flux == {}:
            logging.debug("No beams of interpolated flux found")
            return None

        for aperture, flux in interpolated_flux.iteritems():
            if np.all(~np.isfinite(flux)):
                logging.debug("{0} beam returned all non-finite values for interpolated flux".format(aperture))
                return None

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


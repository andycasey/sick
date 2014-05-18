# coding: utf-8

""" Model class """

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
_scope_interpolator_ = None

def load_model_data(filename, **kwargs):
    """
    Load dispersion or flux data from a filename. 

    Args:
        filename (str): the filename (FITS, memmap, ASCII) to load from. 
    Returns:
        An array of data values.
    """

    # Load by the filename extension
    if filename.endswith(".fits"):
        with pyfits.open(filename, **kwargs) as image:
            data = image[0].data

    elif filename.endswith(".memmap"):
        # Set preferred keyword arguments
        kwargs.setdefault("mode", "c")
        # TODO: Do we still need copy-on-read?
        kwargs.setdefault("dtype", np.double)
        data = np.memmap(filename, **kwargs)

    else:
        # Try ASCII
        data = np.loadtxt(filename, **kwargs)
    
    return data


class Model(object):
    """
    A class to represent the approximate data-generating model for spectra.
    """

    def __init__(self, filename, validate=True):
        """
        Initialise a model class from a filename.

        Args:
            filename (str): a YAML or JSON-style formatted filename.
            validate (bool): validate that the model is specified correctly.
        Raises:
            IOError: if `filename` does not exist.
            ValueError: if there is a mis-match in dispersion and flux points.
            TypeError: if the grid points do not contain column information.
        """

        if not os.path.exists(filename):
            raise IOError("no model filename {0} exists".format(filename))

        parse = yaml.load if filename.endswith(".yaml") else json.load
        
        # Load the model filename
        with open(filename, "r") as fp:
            self.configuration = parse(fp)

        # Perform validation checks to ensure there are no forseeable problems
        if validate:
            self.validate()

        # For legacy purposes, we need to explicitly get a data type
        dtype = np.double if self.configuration["solver"].get("use_double", False) else np.float32

        # Regardless of whether the model is cached or not, the dispersion is specified in
        # the same way: channels -> <channel_name> -> dispersion_filename

        # Load the dispersions
        self.dispersion = dict(zip(self.channels, \
            [load_model_data(self.configuration["channels"][channel]["dispersion_filename"], dtype=dtype) \
            for channel in self.channels]))

        # Is this a cached model?
        if "points_filename" in self.configuration["channels"]:

            # Grid points must be pickled data so that the dimension names are known
            with open(self.configuration["channels"]["points_filename"], "rb") as fp:
                self.grid_points = pickle.load(fp)
                num_points = len(self.grid_points)
                
            if len(self.grid_points.dtype.names) == 0:
                raise TypeError("cached grid points filename has no column names")

            global _scope_interpolator_

            # Do we have a single filename for all fluxes?
            missing_flux_filenames = []
            if "flux_filename" in self.configuration["channels"]:
                
                fluxes = np.memmap(self.configuration["channels"]["flux_filename"], 
                    mode="r+", dtype=dtype).reshape((num_points, -1))

            else:
                # We are expecting flux filenames in each channel (this is less efficient)
                fluxes = []
                for i, channel in enumerate(self.channels):
                    if not "flux_filename" in self.configuration["channels"][channel]:
                        missing_flux_filenames.append(channel)
                        #raise KeyError("no flux filename specified for {} channel".format(channel))
                        continue

                    fluxes.append(np.memmap(self.configuration["channels"][channel]["flux_filename"], 
                        mode="r+", dtype=dtype).reshape((num_points, -1)))

                fluxes = fluxes[0] if len(fluxes) == 1 else np.hstack(fluxes)

            total_flux_pixels = fluxes.shape[1]
            total_dispersion_pixels = sum(map(len, self.dispersion.values()))
            if total_flux_pixels != total_dispersion_pixels:
                for channel in missing_flux_filenames:
                    logger.warn("No flux filename specified for {} channel".format(channel))

                raise ValueError("the total flux pixels ({0}) was different to what was expected ({1})".format(
                    total_flux_pixels, total_dispersion_pixels))

            _scope_interpolator_ = interpolate.LinearNDInterpolator(
                self.grid_points.view(float).reshape((num_points, -1)), fluxes)
            del fluxes

        else:
            # We are expecting dispersion_filename and flux_filenames in each channel

            # Check that there aren't any additional things specified outside of the
            # actual channel names
            assert len(self.configuration["channels"]) == len(self.channels)
            
            self.grid_points = None
            self.flux_filenames = {}

            dimensions = []
            for i, channel in enumerate(self.channels):

                # We will store the filenames and we will load and interpolate on the fly
                folder = self.configuration["channels"][channel]["flux_folder"]
                re_match = self.configuration["channels"][channel]["flux_filename_match"]

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
                    raise ValueError("number of model points found in {0} channel ({1})"
                        " did not match the number in {2} channel ({3})".format(
                        self.channels[0], len(self.grid_points), channel, len(points)))
                       
                # Check the first model flux to ensure it's the same length as the dispersion array
                first_point_flux = load_model_data(matched_filenames[0])
                if len(first_point_flux) != len(self.dispersion[channel]):
                    raise ValueError("number of dispersion ({0}) and flux ({1}) points in {2} "
                        "channel do not match".format(len(self.dispersion[channel]), len(first_point_flux),
                        channel))

                if i == 0:
                    # Save the grid points as a record array
                    self.grid_points = np.core.records.fromrecords(points, names=dimensions,
                        formats=["f8"]*len(dimensions))
                    self.flux_filenames[channel] = matched_filenames

                else:
                    sort_indices = np.argsort(map(self.check_grid_point, points))
                    self.flux_filenames[channel] = [matched_filenames[index] for index in sort_indices]

        # Pre-compute the grid boundaries
        self.grid_boundaries = dict(zip(self.grid_points.dtype.names, [(np.min(self.grid_points[_]), np.max(self.grid_points[_])) \
            for _ in self.grid_points.dtype.names]))
        return None


    def __str__(self):
        return unicode(self).encode("utf-8")


    def __unicode__(self):
        num_channels = len(self.channels)
        num_models = len(self.grid_points) * num_channels
        num_pixels = sum([len(dispersion) * num_models for dispersion in self.dispersion.values()])
        
        return u"{module}.Model({num_models} models; {num_total_parameters} parameters: {num_nuisance_parameters} additional parameters,"\
            " {num_grid_parameters} grid parameters: {parameters}; {num_channels} channels: {channels}; ~{num_pixels} pixels)".format(
            module=self.__module__, num_models=num_models, num_channels=num_channels, channels=', '.join(self.channels),
            num_pixels=human_readable_digit(num_pixels), num_total_parameters=len(self.dimensions),
            num_nuisance_parameters=len(self.dimensions) - len(self.grid_points.dtype.names), 
            num_grid_parameters=len(self.grid_points.dtype.names), parameters=', '.join(self.grid_points.dtype.names))


    def __repr__(self):
        return u"<{0}.Model object with hash {1} at {2}>".format(self.__module__, self.hash[:10], hex(id(self)))


    @property
    def hash(self):
        return md5(json.dumps(self.configuration).encode("utf-8")).hexdigest()

    
    @property
    def channels(self):
        if hasattr(self, "_channels"):
            return self._channels

        protected_channel_names = ("points_filename", "flux_filename")
        return list(set(self.configuration["channels"].keys()).difference(protected_channel_names))


    def map_channels(self, observations):
        """
        Reference model channels to observed spectra based on their dispersions.

        Args:
            observations (list of Spectrum1D objects): The observed spectra.
        Returns:
            mapped_channels (list of str): A list of model channels mapped to each observed channel.
        """

        mapped_channels = []
        for i, spectrum in enumerate(observations):

            # Initialise the list
            channels_found = []

            observed_wlmin = np.min(spectrum.disp)
            observed_wlmax = np.max(spectrum.disp)

            for model_channel, model_dispersion in self.dispersion.iteritems():

                model_wlmin = np.min(model_dispersion)
                model_wlmax = np.max(model_dispersion)

                # Is there overlap?
                if (model_wlmin <= observed_wlmin and observed_wlmax <= model_wlmax) \
                or (observed_wlmin <= model_wlmin and model_wlmax <= observed_wlmax) \
                or (model_wlmin <= observed_wlmin and (observed_wlmin <= model_wlmax and model_wlmax <= observed_wlmax)) \
                or ((observed_wlmin <= model_wlmin and model_wlmin <= observed_wlmax) and observed_wlmax <= model_wlmax):
                    channels_found.append(model_channel)

            if len(channels_found) == 0:
                raise ValueError("no model channels found for observed dispersion map from {wl_start:.1f} to {wl_end:.1f}".format(
                    wl_start=observed_wlmin, wl_end=observed_wlmax))

            elif len(channels_found) > 1:
                index = np.argmin(np.abs(np.mean(spectrum.disp) - map(np.mean, [self.dispersion[channel] for channel in channels_found])))
                channels_found = [channels_found[index]]

                logging.warn("Multiple model channels found for observed channel {0} ({1:.0f} to {2:.0f}). Using '{0}'"
                    " because it's closest by mean dispersion.".format(i, observed_wlmin, observed_wlmax, channels_found[0]))


            mapped_channels.append(channels_found[0])

        # Check that the mean pixel size in the model dispersion maps is smaller than the observed dispersion maps
        for channel, spectrum in zip(mapped_channels, observations):

            mean_observed_pixel_size = np.mean(np.diff(spectrum.disp))
            mean_model_pixel_size = np.mean(np.diff(self.dispersion[channel]))
            if mean_model_pixel_size > mean_observed_pixel_size:
                raise ValueError("the mean model pixel size in the {channel} channel is larger than the mean"
                    " pixel size in the observed dispersion map from {wl_start:.1f} to {wl_end:.1f}".format(
                        channel=channel, wl_start=np.min(spectrum.disp), wl_end=np.max(spectrum.disp)))

        # Keep an internal reference of the channel mapping
        self._channels = mapped_channels
        return mapped_channels


    def validate(self):
        """
        Validate that the model has been specified properly.
        """

        self._validate_channels()
        self._validate_solver()
        self._validate_normalisation()
        self._validate_doppler()
        self._validate_smoothing()
        self._validate_masks()

        return True


    def _validate_normalisation(self):
        """
        Validate that the normalisation settings in the model are specified correctly.

        Returns:
            True if the normalisation settings for this model are specified correctly.
        Raises:
            KeyError: if a model channel does not have a normalisation settings specified.
            TypeError: if an incorrect data type is specified for a normalisation setting.
            ValueError: if an incompatible data value is specified for a normalisation setting.
        """

        if "normalise" not in self.configuration.keys():
            return True

        # Verify the settings for each channel.
        for channel in self.channels:

            # Are there any normalisation settings specified for this channel?
            if not self.configuration["normalise"].get(channel, None): continue

            settings = self.configuration["normalise"][channel]
            if "method" not in settings:
                raise KeyError("configuration setting 'normalise.{}.method' not found".format(channel))

            method = settings["method"]
            if method == "spline":

                knots = settings.get("knots", 0)
                if not isinstance(knots, (int, )):
                    # Could be a list-type of rest wavelength points
                    if not isinstance(knots, (tuple, list, np.ndarray)):
                        raise TypeError("configuration setting 'normalise.{}.knots' is expected"
                            "to be an integer or a list-type of rest wavelength points".format(channel))

                    try: map(float, knots) 
                    except (TypeError, ValueError) as e:
                        raise TypeError("configuration setting 'normalise.{}.knots' is expected"
                            "to be an integer or a list-type of rest wavelength points".format(channel))

            elif method == "polynomial":

                if "order" not in settings:
                    raise KeyError("configuration setting 'normalise.{}.order' not found".format(channel))

                elif not isinstance(settings["order"], (float, int)):
                    raise TypeError("configuration setting 'normalise.{}.order'"
                        " is expected to be an integer-like object".format(channel))

            else:
                raise ValueError("configuration setting 'normalise.{}.method' not recognised"
                    " -- must be spline or polynomial".format(channel))
        
        return True


    def _validate_solver(self):
        """
        Validate that the solver settings in the model are specified correctly.

        Returns:
            True if the solver settings for this model are specified correctly.
        Raises:
            KeyError: if a model channel does not have a normalisation settings specified.
            TypeError: if an incorrect data type is specified for a normalisation setting.
        """

        solver = self.configuration.get("solver", {})
        integer_keys_required = ("sample", "walkers", "burn")
        for key in integer_keys_required:
            if key not in solver:
                raise KeyError("configuration setting 'solver.{}' not found".format(key))

            try: int(solver[key])
            except (ValueError, TypeError) as e:
                raise TypeError("configuration setting 'solver.{}' must be an integer-like type".format(key))

        if solver.get("optimise", True):

            # If we are optimising, then we need initial_samples
            if "initial_samples" not in solver:
                raise KeyError("configuration setting 'solver.initial_samples' is required for optimisation and was not found")

            try: int(solver["initial_samples"])
            except (ValueError, TypeError) as e:
                raise TypeError("configuration setting 'solver.initial_samples' must be an integer-like type")

        if "threads" in solver and not isinstance(solver["threads"], (float, int)):
            raise TypeError("configuration setting 'solver.threads' must be an integer-like type")

        return True


    def _validate_doppler(self):
        """
        Validate that the doppler shift settings in the model are specified correctly.

        Returns:
            True if the doppler settings for this model are specified correctly.
        """

        if "doppler_shift" not in self.configuration.keys():
            return True 

        for channel in self.channels:
            if not self.configuration["doppler_shift"].get(channel, None): continue

            # Perform any additional doppler shift checks here
        return True


    def _validate_smoothing(self):
        """
        Validate that the smoothing settings in the model are specified correctly.

        Returns:
            True if the smoothing settings for this model are specified correctly.
        """ 

        if "convolve" not in self.configuration.keys():
            return True 

        for channel in self.channels:
            if not self.configuration["convolve"].get(channel, None): continue
        return True


    def _validate_channels(self):
        """
        Validate that the channels in the model are specified correctly.

        Returns:
            True if the channels in the model are specified correctly.
        Raises:
            KeyError: if no channels are specified.
            ValueError: if an illegal character is present in any of the channel names.
        """

        if "channels" not in self.configuration.keys():
            raise KeyError("no channels found in model file")

        for channel in self.channels:
            if "." in channel:
                raise ValueError("channel name '{0}' cannot contain a full-stop character".format(channel))
        return True


    def _validate_masks(self):
        """
        Validate that the masks in the model are specified correctly.

        Returns:
            True if the masks in the model are specified correctly.
        Raises:
            TypeError: if the masks are not specified correctly.
        """

        # Masks are optional
        if "masks" not in self.configuration.keys():
            return True

        for channel in self.channels: 
            if not self.configuration["masks"].get(channel, None): continue

            if not isinstance(self.configuration["masks"][channel], (tuple, list)):
                raise TypeError("masks must be a list of regions (e.g., [start, stop])")

            for region in self.configuration["masks"][channel]:
                if not isinstance(region, (list, tuple)) or len(region) != 2:
                    raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")

                try: map(float, region)
                except TypeError:
                    raise TypeError("masks must be a list of regions (e.g., [start, stop]) in Angstroms")
        return True


    @property
    def dimensions(self):
        """
        Return the dimensions for the model.
        """

        if hasattr(self, "_dimensions"):
            return self._dimensions

        dimensions = [] + list(self.grid_points.dtype.names)
        for dimension in self.configuration.keys():
        
            if dimension == "normalise":
                # Append normalisation dimensions for each channel
                for channel in self.channels:
                    if not self.configuration[dimension].get(channel, False): continue

                    method = self.configuration[dimension][channel]["method"]
                    assert method in ("polynomial", "spline")

                    if method == "polynomial":
                        order = self.configuration[dimension][channel]["order"]
                        dimensions.extend(["normalise.{0}.c{1}".format(channel, i) \
                            for i in range(order + 1)])

                    else: # Spline
                        knots = self.configuration[dimension][channel].get("knots", 0)
                        dimensions.extend(["normalise.{0}.k{1}".format(channel, i) \
                            for i in range(knots + 1)])

            elif dimension == "doppler_shift":
                # Check which channels have doppler shifts allowed and add them
                dimensions.extend(["z.{}".format(each) \
                    for each in self.channels if self.configuration[dimension].get(channel, False)])

            elif dimension == "convolve":
                # Check which channels have smoothing allowed and add them
                dimensions.extend(["convolve.{}".format(each) \
                    for each in self.channels if self.configuration[dimension].get(channel, False)])  
        
        # Append jitter
        dimensions.extend(["jitter.{}".format(channel) for channel in self.channels])

        # Cache for future
        setattr(self, "_dimensions", dimensions)
        
        return dimensions


    def cache(self, grid_points_filename, dispersion_filenames, flux_filename,
        wavelengths=None, smoothing_kernels=None, sampling_rate=None, clobber=False):
        """
        Cache the model for faster read access at runtime.

        Args:
            grid_points_filename (str): filename for pickling the model grid points.

            dispersion_filenames (dict): A dictionary containing channels as keys,
                and filenames as values. The dispersion points will be memory-mapped
                to the given filenames.

            flux_filename (str): The flux points will be combined into a single array
                and memory-mapped to the given filename.

            wavelengths (dict): A dictionary containing channels as keys and wavelength
                regions as values. The wavelength regions specify the minimal and
                maximal regions to cache in each channel. By default, the full channel
                will be cached.

            smoothing_kernels (dict): A dictionary containing channels as keys and
                smoothing kernels as values. By default no smoothing is performed.

            sampling_rate (dict): A dictionary containing channels as keys and
                sampling rate (integers) as values.

            clobber (bool): Clobber existing grid point, dispersion and flux filenames
                if they already exist.

        Returns:
            Dictionary containing the current model configuration with the newly
            cached channel parameters. This dictionary can be directly written to a
            new model filename.

        Raises:
            IOError: if clobber is set as False, and any of the grid_points, dispersion,
                or flux filenames already exist.

            TypeError: if invalid smoothing kernels or wavelengths are supplied
        """

        if not clobber:
            filenames = [grid_points_filename, flux_filename] + dispersion_filenames.values()
            filenames_exist = map(os.path.exists, filenames)
            if any(filenames_exist):
                raise IOError("filename {} exists and we've been asked not to clobber it".format(
                    filenames[filenames_exist.index(True)]))

        if not isinstance(smoothing_kernels, dict) and smoothing_kernels is not None:
            raise TypeError("smoothing kernels must be a dictionary-type with channels as keys"\
                " and kernels as values")

        if not isinstance(wavelengths, dict) and wavelengths is not None:
            raise TypeError("wavelengths must be a dictionary-type with channels as keys")

        if sampling_rate is None:
            sampling_rate = dict(zip(self.channels, np.ones(len(self.channels))))

        # We need to know how big our arrays will be
        n_points = len(self.grid_points)

        if wavelengths is None:
            n_pixels = []
            wavelength_indices = {}
            for channel in self.channels:
                pixels = len(self.dispersion[channel])
                n_pixels.append(int(np.ceil(pixels/float(sampling_rate[channel]))))
                wavelength_indices[channel] = np.array([0, pixels])

        else:
            n_pixels = []
            wavelength_indices = {}
            for channel in self.channels:
                start, end = self.dispersion[channel].searchsorted(wavelengths[channel])
                wavelength_indices[channel] = np.array([start, end])

                pixels = end - start 
                n_pixels.append(int(np.ceil(pixels/float(sampling_rate[channel]))))

        # Create the grid points filename
        with open(grid_points_filename, "wb") as fp:
            pickle.dump(self.grid_points, fp)

        # Create empty memmap
        dtype = np.double if self.configuration["solver"].get("use_double", False) else np.float32
        flux = np.memmap(flux_filename, dtype=dtype, mode="w+", shape=(n_points, np.sum(n_pixels)))
        for i in xrange(n_points):

            logger.info("Caching point {0} of {1} ({2:.1f}%)".format(i+1, n_points, 100*(i+1.)/n_points))
            for j, channel in enumerate(self.channels):

                sj, ej = map(int, map(sum, [n_pixels[:j], n_pixels[:j+1]]))

                # Get the flux
                si, ei = wavelength_indices[channel]
                channel_flux = load_model_data(self.flux_filenames[channel][i])

                # Do we need to convolve it first?
                if smoothing_kernels is not None and smoothing_kernels.has_key(channel):
                    sigma = (smoothing_kernels[channel]/(2 * (2*np.log(2))**0.5))/np.mean(np.diff(self.dispersion[channel]))
                    channel_flux = ndimage.gaussian_filter1d(channel_flux, sigma)

                # Do we need to resample?
                flux[i, sj:ej] = channel_flux[si:ei:sampling_rate[channel]]

        # Save the resampled dispersion arrays to disk
        for channel, dispersion_filename in dispersion_filenames.iteritems():
            si, ei = wavelength_indices[channel]

            disp = np.memmap(dispersion_filename, dtype=dtype, mode="w+",
                shape=self.dispersion[channel][si:ei:sampling_rate[channel]].shape)
            disp[:] = np.ascontiguousarray(self.dispersion[channel][si:ei:sampling_rate[channel]],
                dtype=dtype)
            del disp

        # Arrays must be contiguous
        flux[:] = np.ascontiguousarray(flux, dtype=dtype)

        # Now we need to save the flux to disk
        del flux

        cached_model = self.configuration.copy()
        cached_model["channels"] = {
            "grid_points": grid_points_filename,
            "flux_filename": flux_filename,
        }
        cached_model.update(dispersion_filenames)
        return cached_model


    def get_nearest_neighbours(self, point, n=1):
        """
        Return the indices of the nearest `n` neighbours to `point`.

        Args:
            point (list): The point to find neighbours around.
            n (int): The number of neighbours to find on each side, in each dimension.
        Returns:
            indices (np.array): The indices of the nearest neighbours.
        Raises:
            ValueError: if the point is incompatible with the grid shape, or if it is
                outside of the grid boundaries.
        """

        if len(point) != len(self.grid_points.dtype.names):
            raise ValueError("point length ({0}) is incompatible with grid shape ({1})".format(
                len(point), len(self.grid_points.dtype.names)))

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
        """
        Return whether the provided point exists in the model grid.

        Args:
            point (list): The point to find in the model grid.
        Returns:
            index (int): The index of the point in the model grid, if it exists. Otherwise False.
        """

        num_dimensions = len(self.grid_points.dtype.names)
        index = np.all(self.grid_points.view(np.float).reshape((-1, num_dimensions)) == np.array([point]).view(np.float),
            axis=-1)
        return False if not any(index) else np.where(index)[0][0]


    def interpolate_flux(self, point, **kwargs):
        """
        Return interpolated model flux at a given point.

        Args:
            point (list): The point to interpolate a model flux at.
        Returns:
            fluxes (dict): Dictionary of channels (keys) and arrays of interpolated fluxes (values).
        Raises:
            ValueError: when a flux point could not be interpolated (e.g., outside grid boundaries)
        """

        global _scope_interpolator_
        if _scope_interpolator_ is not None:

            interpolated_flux = _scope_interpolator_(*point)
            
            interpolated_fluxes = {}
            num_pixels = map(len, [self.dispersion[channel] for channel in self.channels])
            for i, channel in enumerate(self.channels):
                si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
                interpolated_fluxes[channel] = interpolated_flux[si:ei]
            return interpolated_fluxes

        interpolated_fluxes = {}
        indices = self.get_nearest_neighbours(point)
        for channel in self.channels:
            
            flux = np.zeros((len(indices), len(self.dispersion[channel])))
            flux[:] = np.nan

            # Load the flux points
            for i, index in enumerate(indices):
                channel_flux[i, :] = load_model_data(self.flux_filenames[channel][index])
            
            try:
                interpolated_flux[channel] = interpolate.griddata(self.grid_points[indices],
                    channel_flux, [point], **kwargs).flatten()

            except:
                raise ValueError("could not interpolate flux point -- it is likely outside the grid boundaries")
    

    def masks(self, dispersion_maps, **theta):
        """
        Return pixel masks for the model spectra, given theta.

        Args:
            dispersion_maps (list of array objects): The dispersion maps for each channel (keys).
        Returns:
            None if no masks are specified by the model. If pixel masks are specified,
            a pixel mask dictionary with channels as keys, and arrays as values is returned.
            Zero in arrays indicates pixel was masked, one indicates it was used.
        """

        if not self.configuration.get("masks", False):
            return None

        pixel_masks = {}
        for channel, dispersion_map in zip(self.channels, dispersion_maps):
            if channel not in self.configuration["masks"]:
                masks[channel] = None
            
            else:
                # We are required to build a mask
                mask = np.ones(len(dispersion_map))
                if self.configuration["masks"][channel] is not None:

                    for region in self.configuration["masks"][channel]:
                        if "z.{}".format(channel) in theta:
                            z = theta["z.{}".format(channel)]
                            region = np.array(region) * (1. + z)
                            
                        index_start, index_end = np.searchsorted(dispersion_map, region)
                        pixel_masks[index_start:index_end] = 0
                        
                pixel_masks[channel] = mask
        return pixel_masks


    def __call__(self, observations=None, **theta):
        """
        Return normalised, doppler-shifted, convolved and transformed model fluxes.

        Args:
            observations (list of Spectrum1D objects): The observed data.
        Returns:
            model_spectra (list of Spectrum1D objects): Model spectra for the given theta.
        """

        # Get the grid point and interpolate
        point = [theta[parameter] for parameter in self.grid_points.dtype.names]
        interpolated_flux = self.interpolate_flux(point)

        model_fluxes = []
        check_normalisation = "normalise" in self.configuration.keys()
        
        for channel in self.channels:
                
            # TODO do we actualy need to do a copy?
            model_dispersion = self.dispersion[channel].copy()
            model_flux = interpolated_flux[channel].copy()

            # Any smoothing to apply?
            key = "convolve.{}".format(channel)
            if key in theta:
                profile_sigma = theta[key] / (2.*(2*np.log(2))**0.5)
                true_profile_sigma = profile_sigma / np.mean(np.diff(model_dispersion))
                model_flux = ndimage.gaussian_filter1d(model_flux, true_profile_sigma)

            # Doppler shift the spectra
            key = "z.{0}".format(channel)
            if key in theta:
                z = theta[key]
                # Model dispersion needs to be uniformly sampled in log-wavelength space 
                # before the doppler shift can be applied.
                log_delta = np.diff(model_dispersion).min()
                wl_min, wl_max = model_dispersion.min(), model_dispersion.max()
                log_model_dispersion = (1. + z) * np.exp(np.arange(np.log(wl_min), np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

                # Interpolate flux to log-lambda dispersion
                model_flux = np.interp(log_model_dispersion, model_dispersion, model_flux, left=np.nan, right=np.nan)
                model_dispersion = log_model_dispersion

            # Interpolate model fluxes to observed dispersion map
            if observations is not None:
                index = self.channels.index(channel)
                model_flux = np.interp(observations[index].disp,
                    model_dispersion, model_flux, left=np.nan, right=np.nan)
                model_dispersion = observations[index].disp

            # Apply masks if necessary
            if self.configuration.get("masks", None):
                regions = self.configuration["masks"].get(channel, None)
                if regions:
                    for region in self.configuration["masks"][channel]:
                        if key in theta:
                            z = theta[key]
                            # This is an approximation, but a sufficient one.
                            region = np.array(region) * (1. + z)

                        index_start, index_end = np.searchsorted(model_dispersion, region)
                        model_flux[index_start:index_end] = np.nan

            # Normalise model fluxes to the data
            if check_normalisation and self.configuration["normalise"].get(channel, False):
                
                method = self.configuration["normalise"][channel]["method"]

                if method == "polynomial":

                    # The normalisation coefficients should be in the theta
                    order = self.configuration["normalise"][channel]["order"]
                    coefficients = [theta["normalise.{0}.c{1}".format(channel, i)] \
                        for i in range(order + 1)]
                    model_flux *= np.polyval(coefficients, model_dispersion)
    
                elif method == "spline" and observations is not None:

                    num_knots = self.configuration["normalise"][channel]["knots"]
                    observed_channel = observations[self.channels.index(channel)]
                    
                    # Divide the observed spectrum by the model channel spectrum
                    continuum = observed_channel.flux/model_flux

                    # Fit a spline function to the *finite* continuum points, since the model spectra is interpolated
                    # to all observed pixels (regardless of how much overlap there is)
                    finite = np.isfinite(continuum)
                    knots = [kwarg for kwarg in theta.keys() if kwarg.startswith("normalise.{}.k".format(channel))]
                    tck = interpolate.splrep(observed_channel.disp[finite], continuum[finite],
                        w=1./observed_channel.uncertainty[finite], t=knots)

                    # Scale the model by the continuum function
                    model_flux *= interpolate.splev(model_dispersion, tck)

            model_fluxes.append(model_flux)
        return model_fluxes


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
from utils import human_readable_digit 
from specutils import Spectrum1D

_sick_interpolator_ = None
logger = logging.getLogger(__name__.split(".")[0])

def load_model_data(filename, **kwargs):
    """
    Load dispersion or flux data from a filename. 

    :param filename:
        The filename (formatted as FITS, memmap, or ASCII) to load from.

    :type filename:
        str

    :param kwargs: [optional]
        Keyword arguments that are passed to the loader function (e.g. 
        :class:`pyfits.open`, :class:`numpy.memmap`, or :class:`numpy.loadtxt`
        for FITS, memmap or ASCII formats respectively)

    :returns:
        An array of data values.

    :type numpy.ndarray:
    """

    # Load by the filename extension
    if filename.endswith(".fits"):
        with pyfits.open(filename, **kwargs) as image:
            data = image[0].data

    elif filename.endswith(".memmap"):
        kwargs.setdefault("mode", "c")
        kwargs.setdefault("dtype", np.double)
        data = np.memmap(filename, **kwargs)

    else:
        data = np.loadtxt(filename, **kwargs)
    
    return data


def _cache_model_point(index, filenames, num_pixels, wavelength_indices,
    smoothing_kernels, sampling_rate, mean_dispersion_diffs):

    flux = np.zeros(np.sum(num_pixels))
    for i, (channel, flux_filename) in enumerate(filenames.iteritems()):

        sj, ej = map(int, map(sum, [num_pixels[:j], num_pixels[:j+1]]))

        # Get the flux
        si, ei = wavelength_indices[channel]
        channel_flux = load_model_data(flux_filename)

        # Do we need to convolve it first?
        if smoothing_kernels.has_key(channel):
            sigma = (smoothing_kernels[channel]/(2 * (2*np.log(2))**0.5))\
                /mean_dispersion_diffs[channel]
            channel_flux = ndimage.gaussian_filter1d(channel_flux, sigma)

        # Do we need to resample?
        flux[sj:ej] = channel_flux[si:ei:sampling_rate[channel]]

    return (index, flux)


class Model(object):
    """
    A class to represent the approximate data-generating model for spectra.

    :param filename:
        A YAML- or JSON-formatted model filename.

    :type filename:
        str

    :param validate: [optional]
        Validate that the model is specified correctly.

    :type validate:
        bool
    
    :raises:
        IOError if the ``filename`` does not exist.

        ValueError if the number of model dispersion and flux points are mis-matched.

        TypeError if the model is cached and the grid points filename has no column
        (parameter) names.
    """

    def __init__(self, filename, validate=True):
        
        if not os.path.exists(filename):
            raise IOError("no model filename {0} exists".format(filename))

        parse = yaml.load if filename.endswith(".yaml") else json.load
        
        # Load the model filename
        with open(filename, "r") as fp:
            self.configuration = parse(fp)

        # Perform validation checks to ensure there are no forseeable problems
        if validate:
            self.validate()

        # Regardless of whether the model is cached or not, the dispersion is specified in
        # the same way: channels -> <channel_name> -> dispersion_filename

        # Is this a cached model?
        if self.cached:
            self.dispersion = dict(zip(self.channels, \
                [load_model_data(self.configuration["cached_channels"][channel]["dispersion_filename"], dtype=np.double) \
                for channel in self.channels]))

            # Grid points must be pickled data so that the parameter names are known
            with open(self.configuration["cached_channels"]["points_filename"], "rb") as fp:
                self.grid_points = pickle.load(fp)
                num_points = len(self.grid_points)
                
            if len(self.grid_points.dtype.names) == 0:
                raise TypeError("cached grid points filename has no column names")

            global _sick_interpolator_

            # Do we have a single filename for all fluxes?
            missing_flux_filenames = []
            if "flux_filename" in self.configuration["cached_channels"]:
                
                fluxes = np.memmap(self.configuration["cached_channels"]["flux_filename"], 
                    mode="r+", dtype=np.double).reshape((num_points, -1))

            else:
                # We are expecting flux filenames in each channel (this is less efficient)
                fluxes = []
                for i, channel in enumerate(self.channels):
                    if not "flux_filename" in self.configuration["cached_channels"][channel]:
                        missing_flux_filenames.append(channel)
                        continue

                    fluxes.append(np.memmap(self.configuration["cached_channels"][channel]["flux_filename"], 
                        mode="r+", dtype=np.double).reshape((num_points, -1)))

                fluxes = fluxes[0] if len(fluxes) == 1 else np.hstack(fluxes)

            total_flux_pixels = fluxes.shape[1]
            total_dispersion_pixels = sum(map(len, self.dispersion.values()))
            if total_flux_pixels != total_dispersion_pixels:
                for channel in missing_flux_filenames:
                    logger.warn("No flux filename specified for {0} channel".format(channel))

                raise ValueError("the total flux pixels ({0}) was different to what was expected ({1})".format(
                    total_flux_pixels, total_dispersion_pixels))

            points = self.grid_points.view(float).reshape((num_points, -1))
            _sick_interpolator_ = interpolate.LinearNDInterpolator(points, fluxes)
            del fluxes

        else:
            # Model is not cached. Naughty!
            self.dispersion = dict(zip(self.channels, \
                [load_model_data(self.configuration["channels"][channel]["dispersion_filename"], dtype=np.double) \
                for channel in self.channels]))

            assert len(self.configuration["channels"]) == len(self.channels)
            
            self.grid_points = None
            self.flux_filenames = {}

            parameters = []
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
                        if len(parameters) == 0:
                            parameters = sorted(match.groupdict().keys(), key=re_match.index)
           
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
                    self.grid_points = np.core.records.fromrecords(points, names=parameters,
                        formats=["f8"]*len(parameters))
                    self.flux_filenames[channel] = matched_filenames

                else:
                    sort_indices = np.argsort(map(self.check_grid_point, points))
                    self.flux_filenames[channel] = [matched_filenames[index] for index in sort_indices]

        # Pre-compute the grid boundaries
        self.grid_boundaries = dict(zip(self.grid_points.dtype.names, [(np.min(self.grid_points[_]), np.max(self.grid_points[_])) \
            for _ in self.grid_points.dtype.names]))

        # Initialise the parameters property to avoid nasty fringe cases
        _ = self.parameters
        return None


    def __str__(self):
        return unicode(self).encode("utf-8")


    def __unicode__(self):
        num_channels = len(self.channels)
        num_models = len(self.grid_points) * num_channels
        num_pixels = sum([len(dispersion) * num_models for dispersion in self.dispersion.values()])
        
        return u"{module}.Model({num_models} {is_cached} models; {num_total_parameters} parameters: "\
            "{num_nuisance_parameters} additional parameters, {num_grid_parameters} grid parameters:"\
            " {parameters}; {num_channels} channels: {channels}; ~{num_pixels} pixels)".format(
            module=self.__module__, num_models=num_models, num_channels=num_channels,
            channels=', '.join(self.channels), num_pixels=human_readable_digit(num_pixels),
            num_total_parameters=len(self.parameters), is_cached=["", "cached"][self.cached],
            num_nuisance_parameters=len(self.parameters) - len(self.grid_points.dtype.names), 
            num_grid_parameters=len(self.grid_points.dtype.names),
            parameters=', '.join(self.grid_points.dtype.names))


    def __repr__(self):
        return u"<{0}.Model object with hash {1} at {2}>".format(self.__module__, self.hash[:10], hex(id(self)))


    @property
    def hash(self):
        """ Return a MD5 hash of the JSON-dumped model configuration. """ 
        return md5(json.dumps(self.configuration).encode("utf-8")).hexdigest()


    @property
    def cached(self):
        """ Return whether the model has been cached. """
        return "cached_channels" in self.configuration.keys()


    @property
    def channels(self):
        """ Return the model channels. """

        try:
            return self._channels

        except AttributeError:
            key = ["channels", "cached_channels"][self.cached]
            protected_channel_names = ("points_filename", "flux_filename")
            setattr(self, "_channels", list(set(self.configuration[key].keys()).difference(protected_channel_names)))
            return self._channels


    @property
    def priors(self):
        """
        Return the prior distributions employed for the model.

        :returns:
            Dictionary of parameters (keys) and prior distributions (values)
        """

        try:
            return self._priors

        except AttributeError:

            # Some default priors to apply:
            # uniform between grid boundaries
            # uniform ln(jitter) between -10, 1

            self._priors = {}
            self._priors.update(dict(zip(
                ["f.{0}".format(channel) for channel in self.channels],
                ["uniform(-10, 1)"] * len(self.channels)
            )))
            self._priors.update(dict(zip(self.grid_points.dtype.names,
                ["uniform({0}, {1})".format(*self.grid_boundaries[parameter]) for parameter in self.grid_points.dtype.names]
            )))

            self._priors.update(self.configuration.get("priors", {}))
            return self._priors


    def save(self, filename, clobber=False):
        """
        Save the model configuration.

        :param filename:
            The filename to save the model configuration to.

        :type filename:
            str

        :param clobber: [optional]
            Clobber the filename if it already exists.

        :type clobber:
            bool

        :returns:
            True

        :raises:
            IOError
        """

        if os.path.exists(filename) and not clobber:
            raise IOError("model configuration filename exists and we have been asked not to clobber it")

        dump = yaml.dump if filename.endswith(".yaml") else json.dump
        with open(filename, "w+") as fp:
            dump(self.configuration, fp)

        return True


    def map_channels(self, observations):
        """
        Reference model channels to observed spectra based on their dispersions.

        :param observations:
            The observed spectra.

        :type observations:
            list of :class:`sick.specutils.Spectrum1D` objects

        :returns:
            A list of model channels mapped to each observed channel.
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
                logging.warn("The mean model pixel size in the {channel} channel is larger than the mean" \
                    " pixel size in the observed dispersion map from {wl_start:.1f} to {wl_end:.1f}".format(
                        channel=channel, wl_start=np.min(spectrum.disp), wl_end=np.max(spectrum.disp)))

        # Keep an internal reference of the channel mapping
        self._channels = mapped_channels
        return mapped_channels


    def validate(self):
        """
        Validate that the model has been specified properly.

        :returns:
            True
        """

        self._validate_channels()
        self._validate_settings()
        self._validate_normalisation()
        self._validate_doppler()
        self._validate_smoothing()
        self._validate_masks()

        return True


    def _validate_normalisation(self):
        """
        Validate that the normalisation settings in the model are specified correctly.

        :returns:
            True if the normalisation settings for this model are specified correctly.

        :raises:
            KeyError if a model channel does not have a normalisation settings specified.
            TypeError if an incorrect data type is specified for a normalisation setting.
            ValueError if an incompatible data value is specified for a normalisation setting.
        """

        if "normalise" not in self.configuration.keys():
            return True

        # Verify the settings for each channel.
        for channel in set(self.channels):

            # Are there any normalisation settings specified for this channel?
            if not self.configuration["normalise"].get(channel, None): continue

            settings = self.configuration["normalise"][channel]
            if "method" not in settings:
                raise KeyError("configuration setting 'normalise.{0}.method' not found".format(channel))

            method = settings["method"]
            if method == "spline":

                knots = settings.get("knots", 0)
                if not isinstance(knots, (int, )):
                    # Could be a list-type of rest wavelength points
                    if not isinstance(knots, (tuple, list, np.ndarray)):
                        raise TypeError("configuration setting 'normalise.{0}.knots' is expected"
                            "to be an integer or a list-type of rest wavelength points".format(channel))

                    try: map(float, knots) 
                    except (TypeError, ValueError) as e:
                        raise TypeError("configuration setting 'normalise.{0}.knots' is expected"
                            "to be an integer or a list-type of rest wavelength points".format(channel))

            elif method == "polynomial":

                if "order" not in settings:
                    raise KeyError("configuration setting 'normalise.{0}.order' not found".format(channel))

                elif not isinstance(settings["order"], (float, int)):
                    raise TypeError("configuration setting 'normalise.{0}.order'"
                        " is expected to be an integer-like object".format(channel))

            else:
                raise ValueError("configuration setting 'normalise.{0}.method' not recognised"
                    " -- must be spline or polynomial".format(channel))
        
        return True


    def _validate_settings(self):
        """
        Validate that the settings in the model are specified correctly.

        :returns:
            True if the settings for this model are specified correctly.

        :raises:
            KeyError if a model channel does not have a normalisation settings specified.
            TypeError if an incorrect data type is specified for a normalisation setting.
        """

        settings = self.configuration.get("settings", {})
        integer_keys_required = ("sample", "walkers", "burn")
        for key in integer_keys_required:
            if key not in settings:
                raise KeyError("configuration setting 'settings.{0}' not found".format(key))

            try: int(settings[key])
            except (ValueError, TypeError) as e:
                raise TypeError("configuration setting 'settings.{0}' must be an integer-like type".format(key))

        if settings.get("optimise", True):

            # If we are optimising, then we need initial_samples
            if "initial_samples" not in settings:
                raise KeyError("configuration setting 'settings.initial_samples' is required for optimisation and was not found")

            try: int(settings["initial_samples"])
            except (ValueError, TypeError) as e:
                raise TypeError("configuration setting 'settings.initial_samples' must be an integer-like type")

        if "threads" in settings and not isinstance(settings["threads"], (float, int)):
            raise TypeError("configuration setting 'settings.threads' must be an integer-like type")

        return True


    def _validate_doppler(self):
        """
        Validate that the doppler shift settings in the model are specified correctly.

        :returns:
            True if the doppler settings for this model are specified correctly.
        """

        if "redshift" not in self.configuration.keys():
            return True 

        for channel in set(self.channels):
            if not self.configuration["redshift"].get(channel, None): continue

        return True


    def _validate_smoothing(self):
        """
        Validate that the smoothing settings in the model are specified correctly.

        :returns:
            True if the smoothing settings for this model are specified correctly.
        """ 

        if "convolve" not in self.configuration.keys():
            return True 

        for channel in set(self.channels):
            if not self.configuration["convolve"].get(channel, None): continue
        return True


    def _validate_channels(self):
        """
        Validate that the channels in the model are specified correctly.

        :returns:
            True if the channels in the model are specified correctly.

        :raises:
            KeyError if no channels are specified.
            ValueError if an illegal character is present in any of the channel names.
        """

        if "channels" not in self.configuration.keys() and "cached_channels" not in self.configuration.keys():
            raise KeyError("no channels found in model file")

        for channel in set(self.channels):
            if "." in channel:
                raise ValueError("channel name '{0}' cannot contain a full-stop character".format(channel))
        return True


    def _validate_masks(self):
        """
        Validate that the masks in the model are specified correctly.

        :returns:
            True if the masks in the model are specified correctly.

        :raises:
            TypeError if the masks are not specified correctly.
        """

        # Masks are optional
        if "masks" not in self.configuration.keys():
            return True

        for channel in set(self.channels):
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
    def parameters(self):
        """ Return the model parameters. """

        if hasattr(self, "_parameters"):
            return self._parameters

        parameters = [] + list(self.grid_points.dtype.names)
        for parameter in self.configuration.keys():
        
            if parameter == "normalise":
                # Append normalisation parameters for each channel
                for channel in self.channels:
                    if not self.configuration[parameter].get(channel, False): continue

                    method = self.configuration[parameter][channel]["method"]
                    assert method in ("polynomial", "spline")

                    if method == "polynomial":
                        order = self.configuration[parameter][channel]["order"]
                        parameters.extend(["normalise.{0}.c{1}".format(channel, i) \
                            for i in range(order + 1)])
                        
                    elif method == "spline":
                        knots = self.configuration[parameter][channel].get("knots", 0)
                        parameters.extend(["normalise.{0}.k{1}".format(channel, i) \
                            for i in range(knots)])

            elif parameter == "redshift":
                # Check which channels have doppler shifts allowed and add them
                parameters.extend(["z.{0}".format(each) \
                    for each in self.channels if self.configuration[parameter].get(each, False)])

            elif parameter == "convolve":
                # Check which channels have smoothing allowed and add them
                parameters.extend(["convolve.{0}".format(each) \
                    for each in self.channels if self.configuration[parameter].get(each, False)])  

            elif parameter == "outliers" and self.configuration["outliers"]:
                # Append outlier parameters
                parameters.extend(["Pb", "Vb"])
        
        # Append jitter
        parameters.extend(["f.{0}".format(channel) for channel in self.channels])

        # Cache for future
        setattr(self, "_parameters", parameters)
        
        return parameters


    def cache(self, grid_points_filename, flux_filename, dispersion_filenames=None,
        wavelengths=None, smoothing_kernels=None, sampling_rate=None, clobber=False,
        threads=1):
        """
        Cache the model for faster read access at runtime.

        :param grid_points_filename:
            Filename for pickling the model grid points.

        :type grid_points_filename:
            str

        :param flux_filename:
            The filename to save all the memory-mapped flux points to.

        :type flux_filename:
            str

        :param dispersion_filenames: [optional]
            A dictionary containing channels as keys, and filenames as values. The
            dispersion points will be memory-mapped to the given filenames.

        :type dispersion_filenames:
            dict

        :param wavelengths:
            A dictionary containing channels as keys and wavelength regions as values.
            The wavelength regions specify the minimal and maximal regions to cache in
            each channel. By default, the full channel will be cached.

        :type wavelengths:
            dict

        :param smoothing_kernels:
            A dictionary containing channels as keys and smoothing kernels as values.
            By default no smoothing is performed.

        :type smoothing_kernels:
            dict

        :param sampling_rate:
            A dictionary containing channels as keys and sampling rate (integers) as
            values.

        :type sampling_rate:
            dict

        :param clobber: [optional]
            Clobber existing grid point, dispersion and flux filenames if they exist.

        :type clobber:
            bool

        :param threads: [optional]
            The maximum number of threads to use when caching the model.

        :type threads:
            int

        :returns:
            A dictionary containing the current model configuration with the newly
            cached channel parameters. This dictionary can be directly written to a
            new model filename.

        :raises IOError:
            If ``clobber`` is set as ``False``, and any of the ``grid_points``,
            ``dispersion``, or ``flux_filename``s already exist.

        :raises TypeError:
            If an invalid smoothing kernel or wavelength region is supplied.
        """

        if not clobber:
            filenames = [grid_points_filename, flux_filename]
            if dispersion_filenames is not None:
                filenames.extend(dispersion_filenames.values())
            filenames_exist = map(os.path.exists, filenames)
            if any(filenames_exist):
                raise IOError("filename {0} exists and we've been asked not to"\
                    " clobber it".format(filenames[filenames_exist.index(True)]))

        if not isinstance(smoothing_kernels, dict) and smoothing_kernels is not None:
            raise TypeError("smoothing kernels must be a dictionary-type with "\
                "channels as keys and kernels as values")

        if not isinstance(wavelengths, dict) and wavelengths is not None:
            raise TypeError("wavelengths must be a dictionary-type with channels"\
                " as keys")

        if (sampling_rate is not None or wavelengths is not None) and dispersion_filenames is None:
            raise ValueError("dispersion filename must be provided when the "\
                "sampling rate or wavelength range is restricted")

        if sampling_rate is None:
            sampling_rate = dict(zip(self.channels, np.ones(len(self.channels))))

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
        flux = np.memmap(flux_filename, dtype=np.double, mode="w+",
            shape=(n_points, np.sum(n_pixels)))

        processes = []
        pool = multiprocessing.Pool(threads)
        mean_dispersion_diffs = dict(zip(self.channels,
            [np.mean(np.diff(self.dispersion[each])) for each in self.channels]))

        # Run the caching
        for i in range(n_points):

            logger.info("Caching point {0} of {1} ({2:.1f}%)".format(i+1, 
                n_points, 100*(i+1.)/n_points))

            filenames = dict(zip(self.channels,
                [self.flux_filenames[channel][i] for channel in self.channels]))
            processes.append(pool.apply_async(_cache_model_point,
                args=(i, filenames, n_pixels, wavelength_indices, smoothing_kernels,
                    sampling_rate, mean_dispersion_diffs)))

        # Update the array.
        for process in processes:
            index, flux = process.get()
            flux[index, :] = flux

        # Winter is coming; close the pool
        pool.close()
        pool.join()

        # Save the resampled dispersion arrays to disk
        if dispersion_filenames is not None:
            for channel, dispersion_filename in dispersion_filenames.iteritems():
                si, ei = wavelength_indices[channel]

                disp = np.memmap(dispersion_filename, dtype=np.double, mode="w+",
                    shape=self.dispersion[channel][si:ei:sampling_rate[channel]].shape)
                disp[:] = np.ascontiguousarray(self.dispersion[channel][si:ei:sampling_rate[channel]],
                    dtype=np.double)
                del disp

        flux[:] = np.ascontiguousarray(flux, dtype=np.double)
        del flux

        # Create the cached version 
        cached_model = self.configuration.copy()
        cached_model["cached_channels"] = {
            "grid_points": grid_points_filename,
            "flux_filename": flux_filename,
        }
        if dispersion_filenames is not None:
            cached_model["cached_channels"].update(dispersion_filenames)
        else:
            cached_model["cached_channels"].update(dict(zip(self.channels,
                [cached_model["channels"][each] for each in self.channels])))

        return cached_model


    def get_nearest_neighbours(self, point, n=1):
        """
        Return the indices of the nearest `n` neighbours to `point`.

        :param point:
            The point to find neighbours around.

        :type point:
            list

        :param n: [optional]
            The number of neighbours to find on each side, in each parameter.

        :type n:
            int

        :returns:
            The indices of the nearest neighbours as a :class:`numpy.array`.

        :raises:
            ValueError if the point is incompatible with the grid shape, or if it is
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

        :param point:
            The point to find in the model grid.

        :type point:
            list

        :returns:
            The index of the point in the model grid if it exists, otherwise False.
        """

        num_parameters = len(self.grid_points.dtype.names)
        index = np.all(self.grid_points.view(np.float).reshape((-1, num_parameters)) == np.array([point]).view(np.float),
            axis=-1)
        return False if not any(index) else np.where(index)[0][0]


    def interpolate_flux(self, point, **kwargs):
        """
        Return interpolated model flux at a given point.

        :param point:
            The point to interpolate a model flux at.

        :type point:
            list

        :param kwargs: [optional]
            Other parameters that are directly passed to :class:`scipy.interpolate.griddata`
            if the model has not been cached. Otherwise the kwargs are not used.

        :returns:
            A dictionary of channels (keys) and :class:`numpy.array`s of interpolated
            fluxes (values).

        :raises:
            ValueError when a flux point could not be interpolated (e.g., outside grid
            boundaries).
        """

        global _sick_interpolator_
        if _sick_interpolator_ is not None:
       
            interpolated_flux = _sick_interpolator_(*np.array(point).copy())
            if np.all(~np.isfinite(interpolated_flux)):
                raise ValueError("could not interpolate flux point, as it is likely outside the grid boundaries")    
        
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
                raise ValueError("could not interpolate flux point, as it is likely outside the grid boundaries")
    

    def masks(self, dispersion_maps, **theta):
        """
        Return pixel masks for the model spectra, given theta.

        :param dispersion_maps:
            The dispersion maps for each channel.

        :type dispersion_maps:
            List of :class:`numpy.array` objects.

        :param theta:
            The model parameters :math:`\\Theta`.

        :type theta:
            dict

        :returns:
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
                        if "z.{0}".format(channel) in theta:
                            z = theta["z.{0}".format(channel)]
                            region = np.array(region) * (1. + z)
                            
                        index_start, index_end = np.searchsorted(dispersion_map, region)
                        pixel_masks[index_start:index_end] = 0
                        
                pixel_masks[channel] = mask
        return pixel_masks


    def _continuum(self, channel, obs, model_flux, **theta):
    
        method = self.configuration["normalise"][channel]["method"]

        if method == "polynomial":

            # The normalisation coefficients should be in the theta
            order = self.configuration["normalise"][channel]["order"]
            coefficients = [theta["normalise.{0}.c{1}".format(channel, i)] \
                for i in range(order + 1)]

            return np.polyval(coefficients, obs.disp)

            """ 
            elif method == "spline" and observations is not None:

                num_knots = self.configuration["normalise"][channel]["knots"]
                observed_channel = observations[self.channels.index(channel)]
                        
                # Divide the observed spectrum by the model channel spectrum
                continuum = observed_channel.flux/model_flux

                # Fit a spline function to the *finite* continuum points, since the model spectra is interpolated
                # to all observed pixels (regardless of how much overlap there is)
                finite = np.isfinite(continuum)
                knots = [theta["normalise.{channel}.k{n}".format(channel=channel, n=n)] for n in xrange(num_knots)]
                tck = interpolate.splrep(observed_channel.disp[finite], continuum[finite],
                    w=1./np.sqrt(observed_channel.variance[finite]), t=knots)

                # Scale the model by the continuum function
                return lambda dispersion: interpolate.splev(dispersion, tck)
            """

        else:
            raise NotImplementedError("only polynomial continuums represented at the moment") 



    def __call__(self, observations=None, full_output=False, **theta):
        """
        Return normalised, doppler-shifted, convolved and transformed model fluxes.

        :param observations: [optional]
            The observed data.

        :type observations:
            list of :class:`sick.specutils.Spectrum1D` objects.

        :param full_output: [optional]
            Return the model fluxes and the model continuum for each channel.

        :type full_output:
            bool

        :param theta:
            The model parameters :math:`\\Theta`.

        :type theta:
            dict

        :returns:
            Model spectra for the given theta as a list of :class:`sick.specutils.Spectrum1D`
            objects.
        """

        # Get the grid point and interpolate
        interpolated_flux = self.interpolate_flux(
            [theta[parameter] for parameter in self.grid_points.dtype.names]
        )

        model_fluxes = []
        model_continua = []
        for channel, model_flux in interpolated_flux.iteritems():
                
            model_dispersion = self.dispersion[channel]

            # Any smoothing to apply?
            key = "convolve.{0}".format(channel)
            if key in theta:
                profile_sigma = theta[key] / (2.*(2*np.log(2))**0.5)
                true_profile_sigma = profile_sigma / np.mean(np.diff(model_dispersion))
                model_flux = ndimage.gaussian_filter1d(model_flux, true_profile_sigma)

            # Doppler shift the spectra
            key = "z.{0}".format(channel)
            if key in theta and theta[key] != 0:
                z = theta[key]
                # Model dispersion needs to be uniformly sampled in log-wavelength space 
                # before the doppler shift can be applied
                log_delta = np.diff(model_dispersion).min()
                wl_min, wl_max = model_dispersion.min(), model_dispersion.max()
                log_model_dispersion = np.exp(np.arange(np.log(wl_min), np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

                # Interpolate flux to log-lambda dispersion
                log_model_flux = np.interp(log_model_dispersion, model_dispersion, model_flux, left=np.nan, right=np.nan)
                model_dispersion = log_model_dispersion * (1. + z)
                model_flux = log_model_flux

            # Interpolate model fluxes to observed dispersion map
            if observations is not None:
                index = self.channels.index(channel)
                model_flux = np.interp(observations[index].disp,
                    model_dispersion, model_flux, left=np.nan, right=np.nan)
                model_dispersion = observations[index].disp

            # Apply masks if necessary
            if self.configuration.get("masks", None):
                regions = self.configuration["masks"].get(channel, [])
                for region in regions: 
                    if key in theta:
                        z = theta[key]
                        region = np.array(region) * (1. + z)

                    index_start, index_end = np.searchsorted(model_dispersion, region)
                    model_flux[index_start:index_end] = np.nan

            # Normalise model fluxes to the data
            if "normalise" in self.configuration and self.configuration["normalise"].get(channel, False):

                index = self.channels.index(channel)
                obs = observations[index]

                continuum = self._continuum(channel, obs, model_flux, **theta)
                model_flux *= continuum
                model_continua.append(continuum)

            else:
                model_continua.append(0.)

            model_fluxes.append(model_flux)

        if full_output:
            return (model_fluxes, model_continua)
        return model_fluxes


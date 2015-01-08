# coding: utf-8

""" Model class and inference functions for sick """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["Model"]

import cPickle as pickle
import json
import logging
import multiprocessing
import os
import re
from hashlib import md5
from glob import glob
from time import time

import acor
import emcee
import numpy as np
import pyfits
import yaml
from scipy import interpolate, ndimage, optimize as op

import utils
import inference
import specutils
from validation import validate as model_validate

import matplotlib.pyplot as plt

logger = logging.getLogger("sick")

_sick_interpolator_ = None


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

    _default_priors_ = {}
    _default_configuration_ = {
        "priors": {},
        "normalise": False,
        "mask": None,
        "redshift": False,
        "convolve": False,
        "outliers": False,
        "underestimated_noise": False,
        "settings": {
            "memmap_dtype": "double",
            "threads": 1,
            "optimise": True,
            "burn": 1000,
            "effective_samples_for_convergence": 10,
            "sample_until_converged": True,
            "sample": 1000,
            "proposal_scale": 2,
            "check_convergence_frequency": 1000,
            "rescale_interpolator": False,
            "op_method": "powell",
            "op_gtol": 1e-10,
            "op_maxfun": 5000,
            "op_maxiter": 5000,
            "op_xtol": 1e-4,
            "op_ftol": 1e-4
        },
    }

    def __init__(self, filename, validate=True):

        if not os.path.exists(filename):
            raise IOError("no model filename {0} exists".format(filename))

        parse = yaml.load if filename[-5:].lower() == ".yaml" else json.load
        
        # Load the model filename
        with open(filename, "r") as fp:
            content = parse(fp)

        self.configuration = utils.update_recursively(
            self._default_configuration_.copy(), content)

        assert_naive_behaviour = False
        for key in ("walkers", "sample", "burn"):
            if key not in content.get("settings", {}):
                logger.info("Sampler will continue until ~{0:.0f} effective "\
                    "samples have been made.".format(
                        self.configuration["settings"]["effective_samples_for_convergence"]))
                self.configuration["settings"]["sample_until_converged"] = True
                break

        self._memmap_dtype = self.configuration["settings"]["memmap_dtype"]

        # Regardless of whether the model is cached or not, the dispersion is specified in
        # the same way: channels -> <channel_name> -> dispersion_filename

        if self.cached:

            logger.debug("Loading dispersion maps")
            self.dispersion = {}
            for channel, dispersion_filename in self.configuration["cached_channels"].iteritems():
                if channel not in ("points_filename", "flux_filename"):
                    self.dispersion[channel] = load_model_data(
                        dispersion_filename["dispersion_filename"],
                        dtype=self._memmap_dtype)

            # Grid points must be pickled data so that the parameter names are known
            with open(self.configuration["cached_channels"]["points_filename"], "rb") as fp:
                self.grid_points = pickle.load(fp)
                num_points = self.grid_points.size

            if len(self.grid_points.dtype.names) == 0:
                raise TypeError("cached grid points filename has no column names")

            if validate:
                model_validate(self.configuration, self.channels, self.parameters)

            global _sick_interpolator_

            # Do we have a single filename for all fluxes?
            missing_flux_filenames = []
            if "flux_filename" in self.configuration["cached_channels"]:
                
                logger.debug("Loading cached flux")
                fluxes = load_model_data(
                    self.configuration["cached_channels"]["flux_filename"],
                    dtype=self._memmap_dtype).reshape((num_points, -1))

            else:
                # We are expecting flux filenames in each channel (this is less efficient)
                fluxes = []
                logger.debug("Loading cached fluxes")
                for i, channel in enumerate(self.channels):
                    if not "flux_filename" in self.configuration["cached_channels"][channel]:
                        logger.warn("Missing flux filename for {0} channel".format(channel))
                        missing_flux_filenames.append(channel)
                        continue
                    fluxes.append(load_model_data(
                        self.configuration["cached_channels"][channel]["flux_filename"],
                        dtype=self._memmap_dtype).reshape((num_points, -1)))

                fluxes = fluxes[0] if len(fluxes) == 1 else np.hstack(fluxes)

            total_flux_pixels = fluxes.shape[1]
            total_dispersion_pixels = sum(map(len, self.dispersion.values()))
            if total_flux_pixels != total_dispersion_pixels:
                for channel in missing_flux_filenames:
                    logger.warn("No flux filename specified for {0} channel".format(channel))

                raise ValueError("the total flux pixels ({0}) was different to "\
                    " what was expected ({1})".format(total_flux_pixels,
                        total_dispersion_pixels))

            logger.debug("Reshaping grid points")
            points = self.grid_points.view(float).reshape((num_points, -1))
            logger.debug("Creating interpolator..")
            rescale = self.configuration["settings"]["rescale_interpolator"]
            if rescale:
                logger.debug("Rescaling interpolator to unit cube..")
                _sick_interpolator_ = interpolate.LinearNDInterpolator(
                    points, fluxes, rescale=rescale)

            else:
                _sick_interpolator_ = interpolate.LinearNDInterpolator(
                    points, fluxes)
            del fluxes

        else:
            logger.warn("Model is not cached")

            # Model is not cached. Naughty!
            self.dispersion = {}
            for channel, dispersion_filename in self.configuration["channels"].iteritems():
                if channel not in ("points_filename", "flux_filename"):
                    self.dispersion[channel] = load_model_data(
                        dispersion_filename["dispersion_filename"],
                        dtype=self._memmap_dtype)

            
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
                first_point_flux = load_model_data(matched_filenames[0],
                    dtype=self._memmap_dtype)
                if len(first_point_flux) != len(self.dispersion[channel]):
                    raise ValueError("number of dispersion ({0}) and flux ({1}) points in {2} "
                        "channel do not match".format(len(self.dispersion[channel]),
                            len(first_point_flux), channel))

                if i == 0:
                    # Save the grid points as a record array
                    self.grid_points = np.core.records.fromrecords(points, 
                        names=parameters, formats=["f8"]*len(parameters))

                    # Sorting
                    sort_indices = self.grid_points.argsort(order=parameters)
                    self.grid_points = self.grid_points[sort_indices]
                    self.flux_filenames[channel] = [matched_filenames[index] \
                        for index in sort_indices]

                else:
                    sort_indices = np.argsort(map(self._check_grid_point, points))
                    self.flux_filenames[channel] = [matched_filenames[index] \
                        for index in sort_indices]

            if validate:
                model_validate(self.configuration, self.channels, self.parameters)

        # Get the grid boundaries
        self._unique_grid_points = dict(zip(self.grid_points.dtype.names,
            [np.sort(np.unique(self.grid_points[_])) for _ in self.grid_points.dtype.names]))
        self.grid_boundaries = dict(zip(self.grid_points.dtype.names, 
            [(self.grid_points[_].view(float).min(), self.grid_points[_].view(float).max()) \
                for _ in self.grid_points.dtype.names]))

        self.priors = self._default_priors_.copy()
        # Create default priors for f
        for channel in self.channels:
            if "f.{}".format(channel) in self.parameters:
                self.priors["f.{}".format(channel)] = "uniform(-10,1)"
        self.priors.update(self.configuration["priors"])

        return None


    def save(self, filename, clobber=False, **kwargs):
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
            raise IOError("model configuration filename exists and we have been"\
                " asked not to clobber it")

        dump = yaml.dump if filename[-5:].lower() == ".yaml" else json.dump
        with open(filename, "w+") as fp:
            dump(self.configuration, fp, **kwargs)

        return True


    def __str__(self):
        return unicode(self).encode("utf-8")


    def __unicode__(self):
        num_channels = len(self.channels)
        num_models = len(self.grid_points) * num_channels
        num_pixels = sum([len(d) * num_models for d in self.dispersion.values()])
        
        return u"{module}.Model({num_models} {is_cached} models; "\
            "{num_total_parameters} parameters: {num_extra} "\
            "additional parameters, {num_grid_parameters} grid parameters: "\
            "{parameters}; {num_channels} channels: {channels}; ~{num_pixels} "\
            "pixels)".format(module=self.__module__, num_models=num_models, 
            num_channels=num_channels, channels=', '.join(self.channels), 
            num_pixels=utils.human_readable_digit(num_pixels),
            num_total_parameters=len(self.parameters), 
            is_cached=["", "cached"][self.cached],
            num_extra=len(self.parameters) - len(self.grid_points.dtype.names), 
            num_grid_parameters=len(self.grid_points.dtype.names),
            parameters=', '.join(self.grid_points.dtype.names))


    def __repr__(self):
        return u"<{0}.Model object with hash {1} at {2}>".format(self.__module__,
            self.hash[:10], hex(id(self)))


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
        return sorted(self.dispersion, key=lambda x: x[0])


    def _dictify_theta(self, theta):
        if isinstance(theta, dict):
            return theta
        return dict(zip(self.parameters, theta))


    def _undictify_theta(self, theta):
        if isinstance(theta, dict):
            return np.array([theta[p] for p in self.parameters])
        return theta


    def _chi_sq(self, data, theta):
        """
        Return the standard and reduced chi-squared values for some model 
        parameters theta, given the observed data.

        :param data:
            The observed spectra.

        :type data:
            list of :class:`sick.specutils.Spectrum1D` objects

        :param theta:
            The model parameters theta.

        :type theta:
            dict

        :returns:
            The standard chi-squarde value and the reduced chi-squared value:
            ((data - model)**2/variance)/(num_pixels - num_model_parameters - 1)

        :rtype:
            (float, float)
        """

        model_fluxes = self.__call__(data=data, **self._dictify_theta(theta))
        chi_sq, num_pixels = 0, 0
        for observed, model_flux in zip(data, model_fluxes):
            chi_sq_i = (observed.flux - model_flux)**2 * observed.ivariance
            chi_sq += np.nansum(chi_sq_i)
            num_pixels += np.sum(np.isfinite(chi_sq_i))
        r_chi_sq = chi_sq / (num_pixels - len(self.parameters) - 1)
        if num_pixels == 0:
            logger.warn("No finite pixels found for chi-sq calculation!")
        return (chi_sq, r_chi_sq)


    def _check_grid_point(self, point):
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
        index = np.all(self.grid_points.view(np.float)\
            .reshape((-1, num_parameters)) == np.array([point]).view(np.float),
            axis=-1)
        return False if not any(index) else np.where(index)[0][0]


    @property
    def parameters(self):
        """ Return the model parameters. """

        try:
            return self._parameters

        except AttributeError:
            parameters = [] + list(self.grid_points.dtype.names)

            # Normalisation parameters
            if isinstance(self.configuration["normalise"], dict):
                for channel in self.channels:
                    if channel in self.configuration["normalise"]:
                        order = self.configuration["normalise"][channel]["order"]
                        parameters.extend(["normalise.{0}.c{1}".format(channel, i) \
                            for i in range(order + 1)])
            
            # Redshift parameters
            if isinstance(self.configuration["redshift"], dict):
                # Some/all channels
                parameters.extend(["z.{}".format(c) for c in self.channels \
                    if self.configuration["redshift"][c]])

            elif self.configuration["redshift"] == True:
                # All channels
                parameters.extend(["z.{}".format(c) for c in self.channels])

            # Convolution parameters
            if isinstance(self.configuration["convolve"], dict):
                # Some/all channels
                parameters.extend(["convolve.{}".format(c) for c in self.channels \
                    if self.configuration["convolve"][c]])

            elif self.configuration["convolve"] == True:
                # All channels
                parameters.extend(["convolve.{}".format(c) for c in self.channels])

            # Underestimated variance?
            if isinstance(self.configuration["underestimated_noise"], dict):
                # Some/all channels
                parameters.extend(["f.{}".format(c) for c in self.channels \
                    if self.configuration["underestimated_noise"][c]])

            elif self.configuration["underestimated_noise"] == True:
                # All channels
                parameters.extend(["f.{}".format(c) for c in self.channels])

            # Outliers?
            if self.configuration["outliers"] == True:
                parameters.extend(["Pb", "Vb"])

            # Cache for future
            setattr(self, "_parameters", parameters)
        
        return parameters


    def cache(self, grid_points_filename, flux_filename, dispersion_filenames=None,
        wavelengths=None, smoothing_kernels=None, sampling_rate=None, 
        memmap_dtype=None, clobber=False, threads=1, verbose=False):
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

        :param memmap_dtype: [optional]
            The parameter type to use for caching the grid. If not provided, the
            dtype will be taking from the configuration (settings.mammap_dtype).

        :param clobber: [optional]
            Clobber existing grid point, dispersion and flux filenames if they exist.

        :type clobber:
            bool

        :param threads: [optional]
            The maximum number of threads to use when caching the model.

        :type threads:
            int

        :param verbose: [optional]
            Momentarily change the logging level to INFO so that the progress of
            the caching is announced.

        :type verbose:
            bool

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

        assert grid_points_filename != flux_filename, "Grid points and flux "\
            "filename must be different"

        if dispersion_filenames is not None:
            assert grid_points_filename not in dispersion_filenames.values()
            assert flux_filename not in dispersion_filenames.values()

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

        if smoothing_kernels is None:
            smoothing_kernels = {}

        if memmap_dtype is None:
            memmap_dtype = self.configuration["settings"]["memmap_dtype"]

        if verbose:
            current_level = logger.getEffectiveLevel()
            logger.setLevel(logging.DEBUG)
            logger.info("Logging verbosity temporarily changed to {0} (from {1})".format(
                logging.DEBUG, current_level))

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
        fluxes = np.memmap(flux_filename, dtype=memmap_dtype, mode="w+",
            shape=(n_points, np.sum(n_pixels)))

        processes = []
        mean_dispersion_diffs = dict(zip(self.channels,
            [np.mean(np.diff(self.dispersion[each])) for each in self.channels]))

        if threads > 1:
            pool = multiprocessing.Pool(threads)
        
        # Run the caching
        for i in range(n_points):

            logger.info("Caching point {0} of {1} ({2:.1f}%)".format(i+1, 
                n_points, 100*(i+1.)/n_points))

            filenames = dict(zip(self.channels,
                [self.flux_filenames[channel][i] for channel in self.channels]))

            if threads > 1:
                processes.append(pool.apply_async(_cache_model_point,
                    args=(i, filenames, n_pixels, wavelength_indices, smoothing_kernels,
                        sampling_rate, mean_dispersion_diffs,)))

            else:
                index, flux = _cache_model_point(i, filenames, n_pixels,
                    wavelength_indices, smoothing_kernels, sampling_rate,
                    mean_dispersion_diffs, memmap_dtype)
                fluxes[index, :] = flux

        if threads > 1:
            # Update the array.
            for process in processes:
                index, flux = process.get()
                logger.info("Completed caching point {0}".format(index))
                fluxes[index, :] = flux

            # Winter is coming; close the pool
            pool.close()
            pool.join()

        # Save the resampled dispersion arrays to disk
        if dispersion_filenames is not None:
            for channel, dispersion_filename in dispersion_filenames.iteritems():
                si, ei = wavelength_indices[channel]

                disp = np.memmap(dispersion_filename, dtype=memmap_dtype,
                    mode="w+",
                    shape=self.dispersion[channel][si:ei:sampling_rate[channel]].shape)
                disp[:] = np.ascontiguousarray(self.dispersion[channel][si:ei:sampling_rate[channel]],
                    dtype=np.double)
                del disp

        fluxes[:] = np.ascontiguousarray(fluxes, dtype=np.double)
        del fluxes

        # Create the cached version 
        cached_model = self.configuration.copy()
        cached_model["cached_channels"] = {
            "points_filename": grid_points_filename,
            "flux_filename": flux_filename,
        }
        if dispersion_filenames is not None:
            for channel, dispersion_filename in dispersion_filenames.iteritems():
                cached_model["cached_channels"][channel] = {
                    "dispersion_filename": dispersion_filename
                }
        else:
            for channel in self.channels:
                cached_model["cached_channels"][channel] = {
                    "dispersion_filename": cached_model["channels"][each]
                }
            
        # Return logger back to original levele
        if verbose:
            logger.info("Resetting logging level back to {0}".format(current_level))
            logger.setLevel(current_level)

        return cached_model


    def initial_theta(self, data, **kwargs):
        """
        Return the closest point within the grid that matches the observed data.
        If redshift is modelled, the data are cross-correlated with the entire
        grid and the redshifts are taken from the points with the highest cross-
        correlation function. Normalisation coefficients are calculated by
        matrix inversion such that the best normalisation is performed for each
        possible model point.

        :param data:
            The observed data.

        :type data:
            list of :class:`sick.specutils.Spectrum1D` objects

        :returns:
            The closest point within the grid that can be modelled to fit the
            data, and the approximate reduced chi-sq value for that theta.

        :rtype:
            (dict, float)
        """

        if "initial_theta" in self.configuration:
            missing_parameters = set(self.parameters).difference(self.configuration["initial_theta"].keys())
            if len(missing_parameters) > 0:
                raise KeyError("missing parameter(s) {} for initial_theta".format(", ".join(missing_parameters)))
                    
            logger.info("Using initial theta from model configuration: {0}".format(self.configuration["initial_theta"]))
            full_output = kwargs.pop("full_output", False)
            if full_output:
                return (self.configuration["initial_theta"], np.nan, [])
            return self.configuration["initial_theta"], np.nan

        # Single flux file assumed
        intensities = load_model_data(
            self.configuration["cached_channels"]["flux_filename"],
            dtype=self._memmap_dtype)\
            .reshape((self.grid_points.size, -1))

        readable_point = lambda x: ", ".join(["{0} = {1:.2f}".format(p, v) \
            for p, v in zip(self.grid_points.dtype.names, x)])

        # I only became an astronomer so I could move to the 90210 postcode and
        # call myself a 'Doctor to the Stars'.
        theta = {}
        continuum_coefficients = {}
        chi_sqs = np.zeros(self.grid_points.size)
        num_pixels = [self.dispersion[c].size for c in self.channels]
        for i, (channel, observed) in enumerate(zip(self.channels, data)):
            logger.debug("Calculating initial point in {0} channel..".format(channel))

            # Indices and other information
            dispersion = self.dispersion[channel]
            si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
            logger.debug("Points {0} -> {1} for {2} channel".format(si, ei, channel))
            if isinstance(self.configuration["normalise"], dict):
                continuum_order = self.configuration["normalise"].get(channel, {"order": -1})["order"]
            else:
                continuum_order = -1
            
            # Any redshift to model?
            if self.configuration["redshift"] == True \
            or (isinstance(self.configuration["redshift"], dict) 
                and self.configuration["redshift"].get(channel, False)):

                # Perform cross-correlation against the entire grid!
                logger.debug("Performing cross-correlation in {0} channel..".format(channel))

                z, z_err, R = specutils.cross_correlate_multiple(
                    dispersion, intensities[:, si:ei], observed,
                    continuum_order=continuum_order)

                # Identify and announce the best one
                index = R.argmax()
                logger.debug("Highest CCF in {0} channel was R = {1:.3f} with z = "\
                    "{2:.2e} +/- {3:.2e} ({4:.2f} +/- {5:.2f} km/s) at point {6}".format(
                        channel, R[index], z[index], z_err[index], z[index] * 299792.458,
                        z_err[index] * 299792.458, readable_point(self.grid_points[index])))

                # Store the redshift
                theta["z.{}".format(channel)] = z[index]

                # Apply the redshift
                # Note z is the *measured* redshift, so we need to apply -z to 
                # put the observed spectrum back at rest
                data = np.interp(dispersion, observed.disp * (1. - z[index]),
                    observed.flux, left=np.nan, right=np.nan)
                ivariance = np.interp(dispersion, observed.disp * (1. - z[index]),
                    observed.ivariance, left=np.nan, right=np.nan)

            else:
                # Interpolate the data onto the model grid
                # IF YOU LIKE KITTENS, YOU SHOULD NEVER DO THIS FOR SCIENCE
                # THIS IS JUST FOR AN EXTREMELY ROUGH GUESTIMATE
                data = np.interp(dispersion, observed.disp,
                    observed.flux, left=np.nan, right=np.nan)
                ivariance = np.interp(dispersion, observed.disp,
                    observed.ivariance, left=np.nan, right=np.nan)
            
            # Get only finite data pixels and apply any mask
            finite = np.isfinite(data) * self.mask(dispersion)
            if finite.sum() == 0:
                logger.warn("No finite pixels in the {} data!".format(channel))

            # Any continuum modeling?
            if continuum_order >= 0:
                logger.debug("Performing matrix calculations for {0} channel..".format(channel))
                c = np.ones((continuum_order + 1, finite.sum()))
                for j in range(continuum_order + 1):
                    c[j] *= dispersion[finite]**j

                A = (data[finite]/intensities[:, si:ei][:, finite]).T
                continuum_coefficients[channel] = np.linalg.lstsq(c.T, A)[0]

                continuum = np.dot(continuum_coefficients[channel].T, c)
                model = intensities[:, si:ei][:, finite] * continuum

            else:
                model = intensities[:, si:ei][:, finite]

            # Add to the chi-sq values
            chi_sqs += np.nansum((data[finite] - model)**2 * ivariance[finite],
                axis=1)

            index = chi_sqs.argmin()
            reduced_chi_sq = chi_sqs[index] \
                /(sum(num_pixels[:i+1]) - len(self.grid_points.dtype.names) - 1)
            logger.debug("So far the closest point has r-chi-sq ~ {0:.2f} where"\
                " {1:s}".format(reduced_chi_sq, readable_point(self.grid_points[index])))
        
        # Find the best match, and update the theta dictionary with the values
        index = np.argmin(chi_sqs)
        theta.update(dict(zip(self.grid_points.dtype.names, self.grid_points[index])))
        for channel, coefficients in continuum_coefficients.iteritems():
            for j, value in enumerate(coefficients[::-1, index]):
                theta["normalise.{0}.c{1:.0f}".format(channel, j)] = value
        reduced_chi_sq = chi_sqs[index]/(sum(num_pixels) - len(self.grid_points.dtype.names) - 1)

        _default_init_values = {
            # Let's try and be reasonable.
            "f": np.finfo(np.float).eps,
            "Po": np.finfo(np.float).eps,
            "Vo": np.finfo(np.float).eps
        }
        _default_init_values.update(kwargs.get("default_init_values", {}))
        for parameter in set(self.parameters).difference(theta):

            if parameter in ("Po", "Vo") or parameter[:2] == "f.":
                theta[parameter] = _default_init_values[parameter]

            elif parameter[:9] == "convolve.":
                # Let's assume the grid is slightly higher resolution than the
                # observed data
                theta[parameter] = 0.01

            else:
                raise ValueError("cannot evaluate initialisation rule for the "\
                    "parameter {}".format(parameter))

        # delete the memory-mapped reference to intensities grid
        del intensities

        logger.debug("Initial theta is {0}".format(theta))
        if kwargs.get("full_output", False):
            (self._undictify_theta(theta), reduced_chi_sq, chi_sqs)
        return (self._undictify_theta(theta), reduced_chi_sq)


    def interpolate_flux_locally(self, point, neighbours=2, **kwargs):
        """
        Return interpolated model fluxes at a given point using only the local 
        neighbouring model grid points.

        :param point:
            The point to interpolate model fluxes at.

        :type point:
            :class:`numpy.array`

        :param neighbours: [optional]
            The number of dimension-wise neighbouring elements to use for
            interpolation.

        :type neighbours:
            int

        :returns:
            The interpolated model flux at the given point.

        :raises:
            ValueError when a flux point could not be interpolated (e.g., 
            outside grid boundaries).
        """ 

        # Specify the minimum number of points required for interpolation
        min_points = 2**len(point)
        n_range = [neighbours] if neighbours > 0 else range(1, 1 - neighbours)

        for n in n_range:

            # Identify the nearby points & create a nearby interpolator
            col_indices = np.ones(len(point), dtype=bool)
            indices = np.ones(self.grid_points.size, dtype=bool)
            for i, (p, v) in enumerate(zip(self.grid_points.dtype.names, point)):
                
                differences = self._unique_grid_points[p] - v

                # Get closest points on either side
                left = self._unique_grid_points[p][differences < 0]
                right = self._unique_grid_points[p][differences > 0]
                left_side, right_side = [
                    left[-min([len(left), n])],
                    right[min([len(right), n]) - 1]
                ]
                within_neighbours = (right_side >= self.grid_points[p]) \
                    * (self.grid_points[p] >= left_side)
                indices[~within_neighbours] = False

            if indices.sum() >= min_points: break

        if min_points > indices.sum():
            raise ValueError("not enough points for interpolation (required "\
                "{0:.0f}, found {1:.0f})".format(min_points, indices.sum()))

        size = self.grid_points.size
        point = np.array(point).reshape(len(point), 1)
        p = self.grid_points.view(float).reshape(size, -1)
        if fluxes in kwargs:
            interpolated_flux = interpolate.griddata(
                p[indices, :], fluxes[indices, :], point.T).flatten()    
        else:
            fluxes = load_model_data(
                self.configuration["cached_channels"]["flux_filename"],
                dtype=self._memmap_dtype).reshape(size, -1)
            interpolated_flux = interpolate.griddata(
                p[indices, :], fluxes[indices, :], point.T).flatten()    
            del fluxes

        interpolated_fluxes = {}
        num_pixels = map(len, [self.dispersion[c] for c in self.channels])
        for i, channel in enumerate(self.channels):
            si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
            interpolated_fluxes[channel] = interpolated_flux[si:ei]
        return interpolated_fluxes


    def interpolate_flux_globally(self, point, **kwargs):
        """
        Return interpolated model flux at a given point from an interpolator
        function that uses a Voroni mesh of all grid points.

        :param point:
            The point to interpolate a model flux at.

        :type point:
            :class:`numpy.array`

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
        interpolated_flux = _sick_interpolator_(*point)
        if np.all(~np.isfinite(interpolated_flux)):
            raise ValueError("could not interpolate flux point, as it is likely"\
                " outside the grid boundaries")    
    
        interpolated_fluxes = {}
        num_pixels = map(len, [self.dispersion[c] for c in self.channels])
        for i, channel in enumerate(self.channels):
            si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
            interpolated_fluxes[channel] = interpolated_flux[si:ei]
        return interpolated_fluxes


    def interpolate_flux(self, point, **kwargs):
        """
        Return the interpolated model flux at a given point.

        :param point:
            The point to interpolate a model flux at.

        :type point:
            :class:`numpy.array`

        :returns:
            A dictionary of channels (keys) and :class:`numpy.array`s of interpolated
            fluxes (values).

        :raises:
            ValueError when a flux point could not be interpolated (e.g., outside grid
            boundaries).
        """

        if not self.cached:
            return self.interpolate_flux_locally(point, **kwargs)
        return self.interpolate_flux_globally(point, **kwargs)


    def mask(self, dispersion, z=0):
        """
        Return a rest-frame pixel mask for the dispersion map.

        :param dispersion:
            The dispersion maps for each channel.

        :type dispersion:
            :class:`numpy.array`

        :param z: [optional]
            The redshift to be applied to the pixel mask.

        :returns:
            A pixel mask. Unmasked pixels have True values and masked values
            have False values.

        :rtype:
            :class:`numpy.array`
        """

        if (isinstance(self.configuration["mask"], bool) \
            and self.configuration["mask"] == False) \
        or self.configuration["mask"] is None:
            return np.ones(dispersion.size, dtype=bool)

        mask = np.ones(dispersion.size, dtype=bool)
        for region in self.configuration["mask"]:
            si, ei = np.searchsorted(dispersion, np.array(region) * (1. + z))
            mask[si:ei] = False
        return mask


    def _continuum(self, channel, dispersion, model_flux, **theta):
        # The normalisation coefficients should be in the theta
        order = self.configuration["normalise"][channel]["order"]
        coefficients = [theta["normalise.{0}.c{1}".format(channel, i)] \
            for i in range(order + 1)]

        return np.polyval(coefficients, dispersion)


    def optimise(self, data, initial_theta=None, **kwargs):
        """
        Numerically optimise the log-probability given the data from an
        optionally provided initial point.

        :param data:
            The observed spectra (data).

        :type data:
            list of :class:`sick.specutils.Spectrum1D` objects

        :param initial_theta: [optional]
            The initial point to optimise from. If not provided then the
            initial point will be determined in the method described in
            the sick paper.

        :type initial_theta:
            dict or list

        :param kwargs: [optional]
            Keyword arguments to pass directly to the optimisation call.

        :type kwargs:
            dict

        :returns:
            The numerically optimised point :math:`\Theta_{opt}`.
        """

        if initial_theta is not None:
            p0 = self._undictify_theta(initial_theta)
            assert len(p0) == len(self.parameters), "Initial theta is missing "\
                "parameters."
        else:
            p0, init_r_chi_sq = self.initial_theta(data)

        # Should we optimise all parameters, or keep some fixed?
        fixed = kwargs.pop("fixed", [])

        optimise_parameters = [p for p in self.parameters if p not in fixed]
        logger.info("Optimising parameters: {}".format(", ".join(optimise_parameters)))
        logger.info("Optimising from point:")
        for p, v in zip(self.parameters, p0):
            logger.info("  {0}: {1:.2e}".format(p, v))

        # Which optimisation algorithm should we use?
        method = kwargs.pop("method", self.configuration["settings"]["op_method"])
        default_keys = {
            "bfgs": ("maxfun", "maxiter"),
            "nelder-meade": ("maxfun", "maxiter", "xtol", "ftol"),
            "cg": ("maxiter", ),
            "powell": ("maxiter", "maxfun")
        }
        if method not in default_keys:
            raise ValueError("optimisation method {0} not recognised "\
                "(availiable: {1})".format(method, ", ".join(default_keys.keys())))

        default_kwargs = {}
        for key in default_keys[method]:
            default_kwargs[key] = self.configuration["settings"]["op_" + key]
        [kwargs.setdefault(k, v) for k, v in default_kwargs.iteritems()]

        bounds = []
        for i, parameter in enumerate(optimise_parameters):
            if parameter in self.grid_points.dtype.names:
                indices = np.ones(self.grid_points.size, dtype=bool)
                for j, p in enumerate(self.grid_points.dtype.names):
                    if p != parameter:
                        indices *= (self.grid_points[p] == p0[j])
                bounds.append((
                    self.grid_points[indices][parameter].min(),
                    self.grid_points[indices][parameter].max()
                ))
            else:
                bounds.append((None, None))
        
        if method in ("nelder-meade", "powell", "bfgs", "cg"):
            kwargs.update({
                "full_output": True,
                "disp": False
            })

        
        if method == "bfgs":
            eps = np.nan * np.ones(len(optimise_parameters))
            for i, parameter in enumerate(optimise_parameters):
                if parameter in self.grid_points.dtype.names:
                    eps[i] = 0.01 * np.ptp(self.grid_boundaries[parameter])

                elif parameter[:10] == "normalise.":
                    eps[i] = 1e-6

                elif parameter[:2] == "z.":
                    eps[i] = 1. / 299792.458

                elif parameter[:9] == "convolve.":
                    eps[i] = 0.01

            kwargs.setdefault("epsilon", eps)
            kwargs.setdefault("bounds", bounds)

        elif method == "cg":
            eps = np.nan * np.ones(len(optimise_parameters))
            for i, parameter in enumerate(optimise_parameters):
                if parameter in self.grid_points.dtype.names:
                    eps[i] = 0.1 * np.ptp(self.grid_boundaries[parameter])

                elif parameter[:10] == "normalise.":
                    eps[i] = 1e-8

                elif parameter[:2] == "z.":
                    eps[i] = 0.01 / 299792.458

                elif parameter[:9] == "convolve.":
                    eps[i] = p0[i] * 0.01
            kwargs.setdefault("epsilon", eps)

        logger.debug("Passing keywords to optimiser (these can be passed directly "\
            "from optimise_settings in the model configuration file if using the "\
            "command line interface): {0}".format(kwargs))
        
        # Optimisation
        log_prob_args = (self.parameters, self.priors, self.channels, 
            [self.dispersion[c] for c in self.channels], self.configuration["mask"],
            len(self.grid_points.dtype.names), [s.disp for s in data],
            [s.flux for s in data], [s.variance for s in data],
            [s.ivariance for s in data])

        threads = self.configuration["settings"]["threads"]
        t_init = time()
        def nlp(t):
            # Fill up parameters
            if len(fixed) > 0:
                theta = dict(zip(optimise_parameters, t))
                for p in set(self.parameters).difference(optimise_parameters):
                    theta[p] = p0[self.parameters.index(p)]
                theta = self._undictify_theta(theta)
            else:
                theta = t

            if self.cached:
                nlp = -_log_probability(theta, *log_prob_args)
            else:
                nlp = -inference.log_probability(theta, self, data)
            return nlp

        def finite_nlp(theta):
            prob = nlp(theta)
            if np.isfinite(prob):
                return prob
            return -1e32

        p0_with_fixed = np.array([p0[self.parameters.index(p)] \
            for p in optimise_parameters])

        if method == "nelder-meade":
            p1_with_fixed, fopt, niter, func_calls, warnflag = op.fmin(nlp,
                p0_with_fixed, **kwargs)
            info = {
                "fopt": fopt,
                "niter": niter,
                "func_calls": func_calls,
                "warnflag": warnflag
            }
            messages = [
                "Maximum number of function evaluations ({}) made.".format(func_calls),
                "Maximum number of iterations ({}) reached.".format(niter)
            ]

        elif method == "bfgs":
            p1_with_fixed, fopt, dinfo = op.fmin_l_bfgs_b(nlp, p0_with_fixed,
                **kwargs)
            info = {
                "fopt": fopt,
                "gopt": dinfo["grad"],
                "niter": dinfo["nit"],
                "func_calls": dinfo["funcalls"],
                "warnflag": dinfo["warnflag"],
            }
            messages = [
                "Maximum number of iterations exceeded.",
                dinfo.get("task", "")
            ]

        elif method == "cg":
            p1_with_fixed, fopt, func_calls, grad_calls, warnflag = op.fmin_cg(
                finite_nlp, p0_with_fixed, **kwargs)
            info = {
                "fopt": fopt,
                "func_calls": func_calls,
                "grad_calls": grad_calls,
                "warnflag": warnflag
            }
            messages = [
                "Maximum number of function evaluations ({}) made.".format(func_calls),
                "Maximum number of iterations reached."
            ]

        elif method == "powell":
            p1_with_fixed, fopt, direc, niter, func_calls, warnflag = op.fmin_powell(nlp,
                p0_with_fixed, **kwargs)
            info = {
                "fopt": fopt,
                "direc": direc,
                "niter": niter,
                "func_calls": func_calls,
                "warnflag": warnflag
            }
            messages = [
                "Maximum number of function evaluations.",
                "Maximum number of iterations."
            ]

        # If there were fixed parameters then re-assemble the proper p1
        if len(fixed) > 0:
            theta = dict(zip(optimise_parameters, p1_with_fixed))
            for p in set(self.parameters).difference(optimise_parameters):
                theta[p] = p0[self.parameters.index(p)]
            p1 = self._undictify_theta(theta)
        else:
            p1 = p1_with_fixed

        # Book-keeping
        chi_sq, r_chi_sq = self._chi_sq(data, p1)
        info.update({
            "time_elapsed": time() - t_init,
            "chi_sq": chi_sq,
            "r_chi_sq": r_chi_sq
        })
        if info.get("warnflag", 0) > 0:
            logger.warn("{} Optimised value might be inaccurate.".format(
                messages[info["warnflag"] - 1]))
        return (p1, r_chi_sq, info)


    def walker_widths(self, data, theta):
        """
        Return a list of standard deviations for each model parameter theta to serve
        as the initial standard deviations for the sampler.
        
        :param data:
            The observed data.

        :type data:
            list of :class:`sick.specutils.Spectrum1D` objects

        :param theta:
            The values of the model parameters theta.

        :type theta:
            list

        :returns:
            Initial standard deviations for each of the model parameters.

        :rtype:
            :class:`numpy.array`
        """

        base_env = { 
            "locals": None,
            "globals": None,
            "__name__": None, 
            "__file__": None,
            "__builtins__": None,
            "normal": np.random.normal,
            "uniform": np.random.uniform,
            "abs": abs,
            "np": np
        }

        # Default stretch size is small.
        widths = 1e-4 * np.ones(len(theta))
        theta = self._undictify_theta(theta)
        for i, (parameter, value) in enumerate(zip(self.parameters, theta)):
            # Check to see if there is an explicit walker width distribution for
            # this parameter
            if parameter in self.configuration.get("initial_walker_widths", {}):
                
                # Copy the environment and update with the value
                env = base_env.copy()
                env["x"] = value
                
                if "." in parameter:
                    channel = parameter.split(".")[1]
                    observed_channel = data[self.channels.index(channel)]
                    env["mean_flux"] = np.nanmean(observed_channel.flux)
                    env["mean_variance"] = np.nanmean(observed_channel.variance)
                    env["mean_dispersion"] = np.nanmean(observed_channel.disp)

                # Evaluate the rule
                rule = self.configuration["initial_walker_widths"][parameter]
                try:
                    widths[i] = eval(rule, env)
                except (TypeError, ValueError):
                    logger.exception("Exception in evaluating walker width rule"\
                        " '{0}'. Ignoring rule.".format(rule))
                else:
                    continue

        return widths


    def infer(self, data, p0=None, theta=None, walkers=None, burn=None, sample=None):
        """
        Set up an EnsembleSampler and sample the parameters given the model and data.

        :param data:
            The observed spectra.

        :type data:
            A list of :class:`sick.specutils.Spectrum1D` objects.

        :param model:
            The model class.

        :type model:
            :class:`sick.models.Model`

        :param p0: [optional]
            The starting point for all the walkers. Either `p0` or `theta` must
            be given.

        :type p0: [optional]
            :class:`numpy.ndarray`

        :param theta: [optional]
            The point to initialise all the walkers around. Either `p0` or `theta`
            must be given.

        :ype theta: [optional]
            :class:`numpy.ndarray`

        :param lnprob0: [optional]
            The log posterior probabilities for the walkers at positions given by
            ``p0``. If ``lnprob0`` is None, the initial values are calculated.

        :type lnprob0:
            :class:`numpy.ndarray`

        :param rstate0: [optional]
            The state of the random number generator.

        :param burn: [optional]
            The number of samples to burn. Defaults to the ``burn`` value in
            ``settings`` of ``model.configuration``.

        :type burn:
            int

        :param sample: [optional]
            The number of samples to make from the posterior. Defaults to the 
            ``sample`` value in ``settings`` of ``model.configuration``.

        :type sample:
            int

        :returns:
            A tuple containing the posteriors, sampler, and general information.
        """

        if (p0 is None and theta is None) \
        or (p0 is not None and theta is not None):
            raise ValueError("either p0 or theta must be given")

        if walkers is None:
            walkers = self.configuration["settings"]["walkers"]

        if burn is None:
            burn = self.configuration["settings"]["burn"]
            logger.info("Burning for {0} steps (from settings.burn)".format(burn))
        if sample is None:
            sample = self.configuration["settings"]["sample"]
            logger.info("Sampling for{0} {1} steps (from settings.sample)".format(
                ["", " [at least]"][self.configuration["settings"]["sample_until_converged"]],
                sample))

        if p0 is None:
            # Create p0 from theta
            widths = self.walker_widths(data, theta)
            p0 = utils.sample_ball(theta, widths, walkers)

        else:
            # Ensure the shape is correct
            assert p0.shape[1] == walkers, "p0 shape does not match that expected"

        mean_acceptance_fractions = []
        autocorrelation_time = np.zeros((burn, len(self.parameters)))

        # Initialise the sampler
        proposal_scale = self.configuration["settings"].get("proposal_scale", 2)
        if proposal_scale != 2:
            logger.info("Using non-standard proposal scale of {0:.2f}".format(proposal_scale))

        if self.cached:
            args = (self.parameters, self.priors, self.channels,
                [self.dispersion[c] for c in self.channels], 
                self.configuration["mask"], len(self.grid_points.dtype.names),
                [s.disp for s in data], [s.flux for s in data],
                [s.variance for s in data], [s.ivariance for s in data])
            sampler = emcee.EnsembleSampler(walkers, len(self.parameters),
                _log_probability, args=args, a=proposal_scale,
                threads=self.configuration["settings"]["threads"])

        else:
            sampler = emcee.EnsembleSampler(walkers, len(self.parameters),
                inference.log_probability, args=(self, data), a=proposal_scale,
                threads=self.configuration["settings"]["threads"])
            
        # Start burning
        t_init = time()
        for i, (pos, lnprob, rstate) in enumerate(sampler.sample(p0, iterations=burn)):
            mean_acceptance_fractions.append(np.mean(sampler.acceptance_fraction))
            
            # Announce progress
            logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f},"\
                " highest log probability in last step was {2:.3e}".format(i + 1,
                mean_acceptance_fractions[-1], np.max(sampler.lnprobability[:, i])))
            if mean_acceptance_fractions[-1] in (0, 1):
                raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(
                    mean_acceptance_fractions[i]))

        # Save the chain and calculated log probabilities for later
        chain, lnprobability = sampler.chain, sampler.lnprobability

        logger.info("Resetting chain...")
        sampler.reset()

        total_steps, production_steps = burn, 0
        converged, lnprob, rstate = None, None, None
        max_steps = self.configuration["settings"].get("maximum_steps", -1)
        max_parameter_len = max(map(len, self.parameters))
        effective = self.configuration["settings"]["effective_samples_for_convergence"]
        while True:

            logger.info("Sampling posterior for {0} steps...".format(sample))
            for j, (pos, lnprob, rstate) in enumerate(sampler.sample(pos, 
                lnprob0=lnprob, rstate0=rstate, iterations=sample), start=total_steps):
                mean_acceptance_fractions.append(np.mean(sampler.acceptance_fraction))

                # Announce progress
                logger.info(u"Sampler has finished step {0:.0f} with <a_f> = "\
                    "{1:.3f}, highest log probability in last step was {2:.3e}"\
                    .format(j, mean_acceptance_fractions[-1], 
                        np.max(sampler.lnprobability[:, j - burn])))
                if mean_acceptance_fractions[-1] in (0, 1):
                    raise RuntimeError("mean acceptance fraction is {0:.0f}!"\
                        .format(mean_acceptance_fractions[-1]))

            total_steps += sample
            production_steps = total_steps - burn

            # Estimate auto-correlation times for all parameters.
            logger.info("Auto-correlation times from {0} production steps:"\
                .format(production_steps))
            try:
                acor_times = sampler.acor

            except RuntimeError:
                logger.exception("Auto-correlation times could not be calculated")
                acor_times = [np.inf] * len(self.parameters)

            for k, (p, a_time) in enumerate(zip(self.parameters, acor_times)):
                is_ok = ["", "[OK]"][production_steps >= effective * a_time]
                logger.info("  tau({0}): {1:6.1f} (~{2:3.0f} effective samples)"\
                    " {3}".format(
                    p.rjust(max_parameter_len), a_time, production_steps/a_time,
                        is_ok))

            # Judge convergence.
            logger.info("Checking for convergence...")
            converged = (production_steps >= effective * max(acor_times))
            if converged:
                logger.info("Achievement unlocked: convergence.")
                break

            elif max_steps > 0 and j > max_steps:
                logger.info("Maximum number of steps made!")
                break

            else:
                if self.configuration["settings"]["sample_until_converged"]:
                    sample = self.configuration["settings"]["check_convergence_frequency"]
                    logger.info("Convergence not achieved. Sampling for another"\
                        " {0} steps".format(sample))
                    continue
                else:
                    logger.warn("Convergence may not be achieved!")

                break
                    
        # Concatenate the existing chain and lnprobability with the posterior samples
        chain = np.concatenate([chain, sampler.chain], axis=1)
        lnprobability = np.concatenate([lnprobability, sampler.lnprobability],
            axis=1)

        # Get the ML theta and calculate chi-sq values
        ml_index = np.argmax(lnprobability.reshape(-1))
        ml_values = chain.reshape(-1, len(self.parameters))[ml_index]
        chi_sq, r_chi_sq = self._chi_sq(data, dict(zip(self.parameters, ml_values)))

        # Get the quantiles
        posteriors = {}
        for parameter_name, ml_value, (map_value, quantile_84, quantile_16) \
        in zip(self.parameters, ml_values, 
            map(lambda v: (v[2], v[2]-v[1], v[0]-v[1]),
                zip(*np.percentile(sampler.chain.reshape(-1, len(self.parameters)),
                    [16, 50, 84], axis=0)))):
            posteriors[parameter_name] = (ml_value, quantile_84, quantile_16)

            # Transform redshift posteriors to velocity posteriors
            if parameter_name[:2] == "z.":
                posteriors["v_rad." + parameter_name[2:]] = \
                    list(np.array(posteriors[parameter_name]) * 299792.458)

        info = {
            "chain": chain,
            "lnprobability": lnprobability,
            "walkers": walkers,
            "burn_steps": burn,
            "production_steps": production_steps,
            "converged": converged,
            "mean_acceptance_fractions": np.array(mean_acceptance_fractions),
            "time_elapsed": time() - t_init,
            "chi_sq": chi_sq,
            "reduced_chi_sq": r_chi_sq,
            "autocorrelation_times": dict(zip(self.parameters, acor_times))
        }
        return (posteriors, sampler, info)


    def __call__(self, data=None, full_output=False, **theta):
        """
        Generate some spectra.

        :param data: [optional]
            The observed data.

        :type data:
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

        # Interpolate the flux
        interpolated_flux = self.interpolate_flux(
            [theta[p] for p in self.grid_points.dtype.names])

        model_fluxes = []
        model_continua = []
        for i, channel in enumerate(self.channels):
            
            model_flux = interpolated_flux[channel]
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
                log_model_dispersion = np.exp(np.arange(np.log(wl_min), 
                    np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

                # Interpolate flux to log-lambda dispersion
                log_model_flux = np.interp(log_model_dispersion, model_dispersion, 
                    model_flux, left=np.nan, right=np.nan)
                model_dispersion = log_model_dispersion * (1. + z)
                model_flux = log_model_flux

            # Interpolate model fluxes to observed dispersion map
            if data is not None:
                model_flux = np.interp(data[i].disp,
                    model_dispersion, model_flux, left=np.nan, right=np.nan)
                model_dispersion = data[i].disp

            # Apply mask if necessary
            if self.configuration["mask"] is not None:
                mask = self.mask(model_dispersion, theta.get(key, 0))
                model_flux[~mask] = np.nan

            # Normalise model fluxes to the data?
            if self.configuration["normalise"] \
            and self.configuration["normalise"].get(channel, False):
                continuum = self._continuum(channel, model_dispersion,
                    model_flux, **theta)
                model_flux *= continuum
                model_continua.append(continuum)

            else:
                model_continua.append(0.)
            model_fluxes.append(model_flux)

        if full_output:
            return (model_fluxes, model_continua)
        return model_fluxes


# It's ~15%+ faster if we do things this way instead of refactoring to the
# inference.* functions.
def _log_prior(theta, parameters, priors):

    log_prior = 0
    for parameter, value in zip(parameters, theta):
        if (parameter[:9] == "convolve." and 0 > value) \
        or (parameter == "Pb" and not (1 > value > 0)) \
        or (parameter == "Vb" and 0 > value):
            return -np.inf

        try:
            prior_rule = priors[parameter]

        except KeyError:
            continue

        else:
            f = eval(prior, _prior_eval_env_)
            log_prior += f(value)

    logging.debug("Returning log prior of {0:.2e} for parameters: {1}".format(
        log_prior, ", ".join(["{0} = {1:.2e}".format(name, value) \
            for name, value in zip(parameters, theta)])))
    return log_prior


def _log_likelihood(theta, parameters, channels, model_dispersions,
    mask, n_grid_parameters, observed_dispersions, observed_fluxes, 
    observed_variances, observed_ivariances):

    global _sick_interpolator_
    try:
        interpolated_flux = _sick_interpolator_(*theta[:n_grid_parameters])
    except (ValueError, ):
        print(-np.inf, dict(zip(parameters, theta)))
        return -np.inf

    likelihood, num_finite_pixels = 0, 0
    theta_dict = dict(zip(parameters, theta))
    num_model_pixels = map(len, model_dispersions)
    
    for i, (channel, model_dispersion) \
    in enumerate(zip(channels, model_dispersions)):

        si, ei = map(int, map(sum, [num_model_pixels[:i], num_model_pixels[:i+1]]))
        model_flux = interpolated_flux[si:ei].copy()

        # Convolution
        convolve_key = "convolve.{}".format(channel)
        if convolve_key in theta_dict:
            wavelength_sigma = theta_dict[convolve_key] / 2.3548200450309493
            pixel_sigma = wavelength_sigma / np.mean(np.diff(model_dispersion))
            model_flux = ndimage.gaussian_filter1d(model_flux, pixel_sigma)

        # Redshift
        redshift_key = "z.{}".format(channel)
        if redshift_key in theta_dict and redshift_key != 0:
            z = theta_dict[redshift_key]
            log_delta = np.diff(model_dispersion).min()
            wl_min, wl_max = model_dispersion.min(), model_dispersion.max()
            log_model_dispersion = np.exp(np.arange(np.log(wl_min), 
                np.log(wl_max), np.log(wl_max/(wl_max-log_delta))))

            # Interpolate flux to log-lambda dispersion
            log_model_flux = np.interp(log_model_dispersion, model_dispersion, 
                model_flux, left=np.nan, right=np.nan)
            model_dispersion = log_model_dispersion * (1. + z)
            model_flux = log_model_flux

        else:
            z = 0

        # Interpolate model fluxes to observed dispersion map
        model_flux = np.interp(observed_dispersions[i], model_dispersion,
            model_flux, left=np.nan, right=np.nan)

        # Apply mask
        if mask is not None:
            for region in mask:
                sm, em = np.searchsorted(observed_dispersions[i],
                    np.array(region) * (1. + z))
                model_flux[sm:em] = np.nan

        # Continuum normalise
        if "normalise.{}.c0".format(channel) in theta_dict:
            j, coefficients = 0, []
            while "normalise.{0}.c{1}".format(channel, j) in theta_dict:
                coefficients.append(theta_dict["normalise.{0}.c{1}".format(channel, j)])
                j += 1

            continuum = np.polyval(coefficients, observed_dispersions[i])
            model_flux *= continuum
            
        else:
            continuum = 0.

        # Calculate likelihood
        # Underestimated variance?
        if "f.{}".format(channel) in theta_dict:
            additive_variance = model_flux * np.exp(2. * theta_dict["f.{0}".format(channel)])
            signal_inverse_variance = 1.0/(observed_variances[i] + additive_variance)
        else:
            additive_variance = 0.
            signal_inverse_variance = observed_ivariances[i]

        signal_likelihood = -0.5 * ((observed_fluxes[i] - model_flux)**2 \
            * signal_inverse_variance - np.log(signal_inverse_variance))
        
        # Are we modelling the outliers as well?
        if "Pb" in theta_dict:
            outlier_inverse_variance = 1.0/(theta_dict["Vb"] + observed_variances[i] \
                + additive_variance)
            outlier_likelihood = -0.5 * ((observed_fluxes[i] - continuum)**2 \
                * outlier_inverse_variance - np.log(outlier_inverse_variance))

            Pb = theta_dict["Pb"]
            finite = np.isfinite(outlier_likelihood * signal_likelihood)
            likelihood += np.sum(np.logaddexp(
                np.log(1. - Pb) + signal_likelihood[finite],
                np.log(Pb) + outlier_likelihood[finite]))

        else:
            finite = np.isfinite(signal_likelihood)
            likelihood += np.sum(signal_likelihood[finite])
        num_finite_pixels += finite.sum()

    if likelihood == 0:
        return -np.inf

    logger.debug("Returning log-likelihood of {0:.2e} with {1:.0f} pixels for "\
        "parameters: {2}".format(likelihood, num_finite_pixels, 
        ", ".join(["{0} = {1:.2e}".format(name, value) \
            for name, value in theta_dict.iteritems()])))  

    return likelihood


def _log_probability(theta, parameters, priors, *args):

    prior = _log_prior(theta, parameters, priors)
    if np.isinf(prior):
        return prior
    return prior + _log_likelihood(theta, parameters, *args)


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
    extension = filename.split(os.path.extsep)[-1].lower()
    if extension == "fits":
        with pyfits.open(filename, **kwargs) as image:
            data = image[0].data

    elif extension == "memmap":
        kwargs.setdefault("mode", "c")
        data = np.memmap(filename, **kwargs)

    else:
        data = np.loadtxt(filename, **kwargs)
    return data


def _cache_model_point(index, filenames, num_pixels, wavelength_indices,
    smoothing_kernels, sampling_rate, mean_dispersion_diffs, memmap_dtype):

    logger.debug("Caching point {0}: {1}".format(index, 
        ", ".join(map(os.path.basename, filenames.values()))))

    flux = np.zeros(np.sum(num_pixels))
    for i, (channel, flux_filename) in enumerate(filenames.iteritems()):

        sj, ej = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))

        # Get the flux
        si, ei = wavelength_indices[channel]
        channel_flux = load_model_data(flux_filename, dtype=memmap_dtype)

        # Do we need to convolve it first?
        if channel in smoothing_kernels:
            sigma = (smoothing_kernels[channel]/(2 * (2*np.log(2))**0.5))\
                /mean_dispersion_diffs[channel]
            channel_flux = ndimage.gaussian_filter1d(channel_flux, sigma)

        # Do we need to resample?
        flux[sj:ej] = channel_flux[si:ei:sampling_rate[channel]]

    return (index, flux)

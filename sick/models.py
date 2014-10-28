# coding: utf-8

""" Model class """

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

import emcee
import numpy as np
import pyfits
import yaml
from scipy import interpolate, ndimage, optimize as op

import inference
import utils
import specutils
from validation import validate as model_validate

logger = logging.getLogger("sick")

_sick_interpolator_ = None

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
        kwargs.setdefault("dtype", np.double)
        data = np.memmap(filename, **kwargs)

    else:
        data = np.loadtxt(filename, **kwargs)
    return data



def _cache_model_point(index, filenames, num_pixels, wavelength_indices,
    smoothing_kernels, sampling_rate, mean_dispersion_diffs):

    logger.debug("Caching point {0}: {1}".format(index, 
        ", ".join(map(os.path.basename, filenames.values()))))

    flux = np.zeros(np.sum(num_pixels))
    for i, (channel, flux_filename) in enumerate(filenames.iteritems()):

        sj, ej = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))

        # Get the flux
        si, ei = wavelength_indices[channel]
        channel_flux = load_model_data(flux_filename)

        # Do we need to convolve it first?
        if channel in smoothing_kernels:
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

    _default_priors_ = {}
    _default_configuration_ = {
        "priors": {},
        "normalise": False,
        "masks": None,
        "redshift": False,
        "convolve": False,
        "outliers": False,
        "underestimated_noise": False,
        "settings": {
            "dtype": "double",
            "threads": 1,
            "optimise": True,
            "burn": 5000,
            "sample_until_converged": True,
            "sample": 10000,
            "proposal_scale": 2,
            "check_convergence_frequency": 1000,
            "rescale_interpolator": False,
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
            self.configuration = utils.update_recursively(
                self._default_configuration_.copy(), parse(fp))

        # Regardless of whether the model is cached or not, the dispersion is specified in
        # the same way: channels -> <channel_name> -> dispersion_filename

        if self.cached:

            logger.debug("Loading dispersion maps")
            self.dispersion = {}
            for channel, dispersion_filename in self.configuration["cached_channels"].iteritems():
                if channel not in ("points_filename", "flux_filename"):
                    self.dispersion[channel] = load_model_data(dispersion_filename["dispersion_filename"])

            #self.dispersion = dict(zip(self.channels, 
            #    [load_model_data(self.configuration["cached_channels"][channel]["dispersion_filename"]) \
            #        for channel in self.channels]))

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
                    self.configuration["cached_channels"]["flux_filename"]).reshape((num_points, -1))

            else:
                # We are expecting flux filenames in each channel (this is less efficient)
                fluxes = []
                logger.debug("Loading cached fluxes")
                for i, channel in enumerate(self.channels):
                    if not "flux_filename" in self.configuration["cached_channels"][channel]:
                        logger.warn("Missing flux filename for {0} channel".format(channel))
                        missing_flux_filenames.append(channel)
                        continue
                    fluxes.append(load_model_data(self.configuration["cached_channels"][channel]["flux_filename"]).reshape((num_points, -1)))

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
                    self.dispersion[channel] = load_model_data(dispersion_filename["dispersion_filename"])

            
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
        self.grid_boundaries = dict(zip(self.grid_points.dtype.names, 
            [(self.grid_points[_].view(float).min(), self.grid_points[_].view(float).max()) \
                for _ in self.grid_points.dtype.names]))

        self.priors = self._default_priors_.copy()
        # Set some implicit priors
        for parameter, ranges in self.grid_boundaries.iteritems():
            self.priors[parameter] = "uniform({0},{1})".format(*ranges)

        # Update with explicit priors
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


    def _chi_sq(self, observations, theta):
        """
        Return the standard and reduced chi-squared values for some model 
        parameters theta, given the observed data.

        :param observations:
            The observed spectra.

        :type observations:
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

        model_fluxes = self.__call__(observations=observations, **self._dictify_theta(theta))
        chi_sq, num_pixels = 0, 0
        for observed, model_flux in zip(observations, model_fluxes):
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
        wavelengths=None, smoothing_kernels=None, sampling_rate=None, clobber=False,
        threads=1, verbose=False):
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
        fluxes = np.memmap(flux_filename, dtype=np.double, mode="w+",
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
                        sampling_rate, mean_dispersion_diffs)))

            else:
                index, flux = _cache_model_point(i, filenames, n_pixels,
                    wavelength_indices, smoothing_kernels, sampling_rate,
                    mean_dispersion_diffs)
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

                disp = np.memmap(dispersion_filename, dtype=np.double, mode="w+",
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


    def initial_theta(self, observations, **kwargs):
        """
        Return the closest point within the grid that matches the observed data.
        If redshift is modelled, the data are cross-correlated with the entire
        grid and the redshifts are taken from the points with the highest cross-
        correlation function. Normalisation coefficients are calculated by
        matrix inversion such that the best normalisation is performed for each
        possible model point.

        :param observations:
            The observed data.

        :type observations:
            list of :class:`sick.specutils.Spectrum1D` objects

        :returns:
            The closest point within the grid that can be modelled to fit the
            data, and the approximate reduced chi-sq value for that theta.

        :rtype:
            (dict, float)
        """

        # Single flux file assumed
        intensities = load_model_data(self.configuration["cached_channels"]["flux_filename"])\
            .reshape((self.grid_points.size, -1))

        readable_point = lambda x: ", ".join(["{0} = {1:.2f}".format(p, v) \
            for p, v in zip(self.grid_points.dtype.names, x)])

        # I only became an astronomer so I could move to the 90210 postcode and
        # call myself a 'Doctor to the Stars'.
        theta = {}
        continuum_coefficients = {}
        chi_sqs = np.zeros(self.grid_points.size)
        num_pixels = [self.dispersion[c].size for c in self.channels]
        for i, (channel, observed) in enumerate(zip(self.channels, observations)):
            logger.debug("Calculating initial point in {0} channel..".format(channel))

            # Indices and other information
            dispersion = self.dispersion[channel]
            si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
            logger.debug("Points {0} -> {1} for {2} channel".format(si, ei, channel))
            continuum_order = self.configuration["normalise"].get(channel, {"order": -1})["order"]
            
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
            
            # Get only finite data pixels
            finite = np.isfinite(data)
            finite[-1] = False
            if finite.sum() == 0:
                logger.warn("No finite pixels in the {} data!".format(channel))

            # Apply mask?
            logger.warn("no masks used for init point")

            # Any normalisation to model?
            if continuum_order >= 0:
                logger.debug("Performing matrix calculations for {0} channel..".format(channel))
                c = np.ones((continuum_order + 1, finite.sum()))
                for j in range(continuum_order + 1):
                    c[j] *= dispersion[finite]**j

                # [f0 f1 f2 f3 f4 f5] = [m1 m2 m3 m3 m4][[b], [b], [b], [b], [b]]
                # [m1 m2 m3 m4 m5]^(-1)[f0 f1 f2 f3 f4 f5] = [[b], [b], [b], [b], [b]]

                # [[f0,0 f1,0 f2,0 f3,0 f4,0] = [b]

                # [[f0, f1, f2, f3, f4] = [[m00, m01, m02, m03, m04] * [[b0],]
                #                         [m10, m11, m12, m13, m14]    [b1],
                #                         [m20, m21, m22, m23, m24]    [b2], 
                A = (data[finite]/intensities[:, si:ei][:, finite]).T
                #B = np.dot(np.matrix(intensities[:, si:ei][:, finite]).I.T, data)
                
                continuum_coefficients[channel] = np.linalg.lstsq(c.T, A)[0] # (continuum_order + 1, N)

                #raise a
                #if np.isfinite(continuum_coefficients[channel]).sum() == 0:
                #    raise a

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
                # Solve for the convolution parameter as a single value.
                # compare convolve, projected flux with observed data
                theta[parameter] = 0.
                logger.warn("NO SMOOTHING VALUE ESTIMATED")

            else:
                raise ValueError("cannot evaluate initialisation rule for the "\
                    "parameter {}".format(parameter))

        # delete the memory-mapped reference to intensities grid
        del intensities

        logger.debug("Initial theta is {0}".format(theta))
        return (self._undictify_theta(theta), reduced_chi_sq)


    def interpolate_local_flux(self, point, neighbours=1):

        # Identify the nearby points & create a nearby interpolator
        col_indices = np.ones(len(point), dtype=bool)
        indices = np.ones(self.grid_points.size, dtype=bool)
        for i, (parameter, value) in enumerate(zip(self.grid_points.dtype.names, point)):

            #print(indices.sum())
            #print(set(self.grid_points[parameter][indices]))

            # Is this point within the grid?
            """
            exact_grid_match = np.less_equal(
                np.abs(self.grid_points[parameter] - value), np.finfo(float).eps)
            if exact_grid_match.any():
                indices[~exact_grid_match] = False
                print("matching all {0} at {1}".format(parameter, value))
                col_indices[i] = False
                print("after", set(self.grid_points[parameter][indices]))
                raise a
                continue
            """

            # Check within some range
            unique_points = np.sort(np.unique(self.grid_points[parameter]))
            differences = unique_points - value

            # Get closest points on either side
            left = unique_points[differences < 0]
            right = unique_points[differences > 0]
            left_side, right_side = [
                left[-min([len(left), neighbours])],
                [neighbours - 1]
            ]
            within_neighbours = (right_side >= self.grid_points[parameter]) \
                * (self.grid_points[parameter] >= left_side)
            indices[~within_neighbours] = False

            #print("after", set(self.grid_points[parameter][indices]))

        # Prevent Qhull errors

        size = self.grid_points.size
        point = np.array(point).reshape(len(point), 1)
        p = self.grid_points.view(float).reshape(size, -1)
        fluxes = load_model_data(
            self.configuration["cached_channels"]["flux_filename"]).reshape(size, -1)
        interpolated_flux = interpolate.griddata(
            p[indices, :], fluxes[indices, :], point.T).flatten()
        del fluxes

        return interpolated_flux


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

        if not self.cached:
            raise TypeError("model must be cached first")

        global _sick_interpolator_
        interpolated_flux = _sick_interpolator_(*np.array(point).copy())
        if np.all(~np.isfinite(interpolated_flux)):
            raise ValueError("could not interpolate flux point, as it is likely"\
                " outside the grid boundaries")    
    
        interpolated_fluxes = {}
        num_pixels = map(len, [self.dispersion[c] for c in self.channels])
        for i, channel in enumerate(self.channels):
            si, ei = map(int, map(sum, [num_pixels[:i], num_pixels[:i+1]]))
            interpolated_fluxes[channel] = interpolated_flux[si:ei]
        return interpolated_fluxes


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

        if not self.configuration["masks"]:
            return None

        pixel_masks = {}
        for channel, dispersion_map in zip(self.channels, dispersion_maps):
            if channel not in self.configuration["masks"]:
                pixel_masks[channel] = None
            
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


    def _continuum(self, channel, observations, model_flux, **theta):
        # The normalisation coefficients should be in the theta
        order = self.configuration["normalise"][channel]["order"]
        coefficients = [theta["normalise.{0}.c{1}".format(channel, i)] \
            for i in range(order + 1)]

        return np.polyval(coefficients, observations.disp)


    def optimise(self, observations, initial_theta=None, **kwargs):
        """
        Numerically optimise the log-probability given the data from an
        optionally provided initial point.

        :param observations:
            The observed spectra (data).

        :type observations:
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
            p0, init_r_chi_sq = self.initial_theta(observations)

        logger.info("Optimising from point:")
        for p, v in zip(self.parameters, p0):
            logger.info("  {0}: {1:.2e}".format(p, v))

        # Set some keyword defaults
        default_kwargs = {
            "maxfun": self.configuration["settings"]["op_maxfun"],
            "maxiter": self.configuration["settings"]["op_maxiter"],
            "xtol": self.configuration["settings"]["op_xtol"],
            "ftol": self.configuration["settings"]["op_ftol"],
            "disp": False
        }
        [kwargs.setdefault(k, v) for k, v in default_kwargs.iteritems()]
        
        # And we need to overwrite this one because we want all the information, 
        # even if the user doesn't.
        kwargs.update({"full_output": True})
        logger.debug("Passing keywords to optimiser (these can be passed directly "\
            "from optimise_settings in the model configuration file if using the "\
            "command line interface): {0}".format(kwargs))

        # Optimisation
        t_init = time()
        nlp = lambda t, m, o: -inference.log_probability(t, m, o)
        p1, fopt, niter, funcalls, warnflag = op.fmin(nlp, p0,
            args=(self, observations), **kwargs)

        # Book-keeping
        chi_sq, r_chi_sq = self._chi_sq(observations, p1)
        info = {
            "fopt": fopt,
            "niter": niter,
            "funcalls": funcalls,
            "warnflag": warnflag,
            "time_elapsed": time() - t_init,
            "chi_sq": chi_sq,
            "r_chi_sq": r_chi_sq
        }
        if warnflag > 0:
            m = [
                "Maximum number of function evaluations ({}) made.".format(funcalls),
                "Maximum number of iterations ({}) reached.".format(niter)
            ][warnflag - 1]
            logger.warn("{0} Optimised solution may be inaccurate.".format(m))

        return (p1, r_chi_sq, info)


    def walker_widths(self, observations, theta):
        """
        Return a list of standard deviations for each model parameter theta to serve
        as the initial standard deviations for the sampler.
        
        :param observations:
            The observed data.

        :type observations:
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
            "abs": abs
        }

        widths = np.zeros(len(theta))
        theta = self._undictify_theta(theta)
        for i, (parameter, value) in enumerate(zip(self.parameters, theta)):

            # Check to see if there is an explicit walker width distribution for
            # this parameter
            if parameter in self.configuration.get("initial_walker_widths", {}):
                
                # Copy the environment and update with the value
                env = base_env.copy()
                env["x"] = value

                # Evaluate the rule
                rule = self.configuration["initial_walker_widths"][parameter]
                try:
                    widths[i] = eval(rule, env)
                except (TypeError, ValueError):
                    logger.exception("Exception in evaluating walker width rule"\
                        " '{0}'. Ignoring rule.".format(rule))
                else:
                    # OK, we have updated the width. Continue to the next param
                    continue

            if parameter in self.grid_points.dtype.names:
                widths[i] = 0.05 * np.ptp(self.grid_boundaries[parameter])

            elif parameter[:2] == "z.": # Redshift
                # Set the velocity width to be 1 km/s
                widths[i] = 1./299792458e-3

            elif parameter[:2] == "f.": # Jitter
                widths[i] = 0.1

            elif parameter[:9] == "convolve.": # Smoothing
                # 10% of the value
                widths[i] = 0.1 * value

            elif parameter[:10] == "normalise.": # Normalisation
                channel, coefficient = parameter.split(".")[1:]
                coefficient = int(coefficient[1:]) # 'normalise.blue.c0'

                observed = observations[self.channels.index(channel)]
                order = self.configuration["normalise"][channel]["order"]

                # And we arbitrarily specify the width to be ~3x the mean uncertainty in flux.
                scale = 3 * np.nanmean(observed.variance)
                widths[i] = scale/(observed.disp.mean()**(order - coefficient))

            elif parameter == "Pb":
                widths[i] = 0.01

            elif parameter == "Vb":
                logger.warn('probs not optimally effective')
                widths[i] = 1e-4

            else:
                raise RuntimeError("whoops")

        return widths


    def infer(self, observed_spectra, p0, burn=None, sample=None):
        """
        Set up an EnsembleSampler and sample the parameters given the model and data.

        :param observed_spectra:
            The observed spectra.

        :type observed_spectra:
            A list of :class:`sick.specutils.Spectrum1D` objects.

        :param model:
            The model class.

        :type model:
            :class:`sick.models.Model`

        :param p0:
            The starting point for all the walkers.

        :type p0:
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

        # Set up MCMC settings and arrays
        walkers = self.configuration["settings"]["walkers"]
        if burn is None:
            burn = self.configuration["settings"]["burn"]
            logger.info("Burning for {0} steps (from settings.burn)".format(burn))
        if sample is None:
            sample = self.configuration["settings"]["sample"]
            logger.info("Sampling for {0} steps (from settings.sample)".format(sample))

        mean_acceptance_fractions = []
        autocorrelation_time = np.zeros((burn, len(self.parameters)))

        # Initialise the sampler
        proposal_scale = self.configuration["settings"].get("proposal_scale", 2)
        if proposal_scale != 2:
            logger.info("Using non-standard proposal scale of {0:.2f}".format(proposal_scale))

        sampler = emcee.EnsembleSampler(walkers, len(self.parameters),
            inference.log_probability, args=(self, observed_spectra),
            threads=self.configuration["settings"]["threads"], a=proposal_scale)

        # Start burning
        t_init = time()
        for i, (pos, lnprob, rstate) in enumerate(sampler.sample(p0, iterations=burn)):
            mean_acceptance_fractions.append(np.mean(sampler.acceptance_fraction))
            
            # Announce progress
            logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f},"\
                " maximum log probability in last step was {2:.3e}".format(i + 1,
                mean_acceptance_fractions[-1], np.max(sampler.lnprobability[:, i])))
            if mean_acceptance_fractions[-1] in (0, 1):
                raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(
                    mean_acceptance_fractions[i]))

        # Save the chain and calculated log probabilities for later
        chain, lnprobability = sampler.chain, sampler.lnprobability

        logger.info("Resetting chain...")
        sampler.reset()

        converged = None
        while True:
            logger.info("Sampling posterior for {0} steps...".format(sample))
            for j, state in enumerate(sampler.sample(pos, iterations=sample)):
                mean_acceptance_fractions.append(np.mean(sampler.acceptance_fraction))

                # Announce progress
                logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f},"\
                    " maximum log probability in last step was {2:.3e}".format(j + i + 2,
                    mean_acceptance_fractions[-1], np.max(sampler.lnprobability[:, j])))
                if mean_acceptance_fractions[-1] in (0, 1):
                    raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(
                        mean_acceptance_fractions[i]))

            # Estimate auto-correlation times for all parameters.
            logger.info("Auto-correlation lengths:")
            acor_lengths = []
            max_parameter_len = max(map(len, self.parameters))
            for i, parameter in enumerate(self.parameters):
                try:
                    acor_time = acor.acor(np.mean(sampler.chain[:, :, i], axis=0))
                except:
                    logger.exception("Error in calculating auto-correlation length "\
                        "for parameter {}:".format(parameter))
                    acor_time = [np.inf]

                is_ok = ["", "[OK]"][sample >= 100 * acor_time[0]]
                logger.info("  acor({0}): {1:.2f}   {2}".format(
                    parameter.rjust(max_parameter_len), acor_time[0], is_ok))
                acor_lengths.append(acor_time[0])

            else:
                # Judge convergence.
                logger.info("Checking for convergence...")
                converged = (sample >= 100 * max(acor_lengths))

                if converged:
                    logger.info("Achievement unlocked: convergence.")
                    break

                else:
                    logger.warn("Convergence may not be achieved.")
                    break
                    
        # Concatenate the existing chain and lnprobability with the posterior samples
        chain = np.concatenate([chain, sampler.chain], axis=1)
        lnprobability = np.concatenate([lnprobability, sampler.lnprobability], axis=1)

        # Get the ML theta and calculate chi-sq values
        ml_index = np.argmax(lnprobability.reshape(-1))
        ml_values = chain.reshape(-1, len(self.parameters))[ml_index]
        chi_sq, r_chi_sq = self._chi_sq(observed_spectra, dict(zip(self.parameters, ml_values)))

        # Get the quantiles
        posteriors = {}
        for parameter_name, ml_value, (map_value, quantile_84, quantile_16) \
        in zip(self.parameters, ml_values, 
            map(lambda v: (v[2], v[2]-v[1], v[0]-v[1]),
                zip(*np.percentile(sampler.chain.reshape(-1, len(self.parameters)), [16, 50, 84], axis=0)))):
            posteriors[parameter_name] = (ml_value, quantile_84, quantile_16)

            # Transform redshift posteriors to velocity posteriors
            if parameter_name[:2] == "z.":
                posteriors["v_rad." + parameter_name[2:]] = list(np.array(posteriors[parameter_name]) * 299792458e-3)

        # Send back additional information
        info = {
            "chain": chain,
            "lnprobability": lnprobability,
            "converged": converged,
            "mean_acceptance_fractions": np.array(mean_acceptance_fractions),
            "time_elapsed": time() - t_init,
            "chi_sq": chi_sq,
            "reduced_chi_sq": r_chi_sq,
            "autocorrelation_lengths": dict(zip(self.parameters, acor_lengths))
        }
        return (posteriors, sampler, info)



    def __call__(self, observations=None, full_output=False, **theta):
        """
        Generate some spectra.

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
            if observations is not None:
                model_flux = np.interp(observations[i].disp,
                    model_dispersion, model_flux, left=np.nan, right=np.nan)
                model_dispersion = observations[i].disp

            # Apply masks if necessary
            if self.configuration["masks"] is not None:
                regions = self.configuration["masks"].get(channel, [])
                for region in regions: 
                    if key in theta:
                        z = theta[key]
                        region = np.array(region) * (1. + z)
                    index_start, index_end = np.searchsorted(model_dispersion, region)
                    model_flux[index_start:index_end] = np.nan

            # Normalise model fluxes to the data?
            if self.configuration["normalise"] \
            and self.configuration["normalise"].get(channel, False):
                obs = observations[i]

                continuum = self._continuum(channel, obs, model_flux, **theta)
                model_flux *= continuum
                model_continua.append(continuum)

            else:
                model_continua.append(0.)
            model_fluxes.append(model_flux)

        if full_output:
            return (model_fluxes, model_continua)
        return model_fluxes
#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Base model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import os
import sys
import yaml
from hashlib import md5
from time import strftime

import numpy as np
from astropy.constants import c as speed_of_light
from scipy.ndimage import gaussian_filter1d

# sick
from .. import specutils
import utils

logger = logging.getLogger("sick")


class BaseModel(object):
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

        ValueError if the number of model wavelength and flux points are 
        different.

        TypeError if the model is cached and the grid points filename has no 
        column (parameter) names.
    """

    _default_configuration = {}

    def __init__(self, filename, validate=True):

        if not os.path.exists(filename):
            raise IOError("no model filename {0} exists".format(filename))

        with open(filename, "r") as fp:
            content = yaml.load(fp)

        self._initialised = False
        self._configuration = utils.update_recursively(
            self._default_configuration.copy(), content)

        # Load the grid points and the metadata.
        with open(self._configuration["model_grid"]["grid_points"], "rb") as fp:
            self.grid_points, self.meta = pickle.load(fp)

        # Set the channel names.
        self.channel_names = self.meta["channel_names"]
        
        # Load the wavelength maps.
        self.wavelengths = np.memmap(
            self._configuration["model_grid"]["wavelengths"], mode="r",
            dtype="float32")
 

    def save(self, filename, clobber=False, **kwargs):
        """
        Save the model configuration to a YAML-formatted file.

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
            raise IOError("filename {} already exists".format(filename))

        kwds = {
            "allow_unicode": True, 
            "default_flow_style": False
        }
        kwds.update(kwargs)
        with open(filename, "w+") as fp:
            yaml.safe_dump(self._configuration, stream=fp, **kwds)
        return True


    def __str__(self):
        return unicode(self).encode("utf-8")


    def __unicode__(self):
        return u"{module}.Model()".format(module=self.__module__)

        """
        num_channels = len(self.channels)
        num_models = len(self.grid_points) * num_channels
        num_pixels = sum([len(d) * num_models for d in self.wavelength.values()])
        
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
        """


    def __repr__(self):
        return u"<{0}.Model object with hash {1} at {2}>".format(
            self.__module__, self.hash[:10], hex(id(self)))

    @property
    def _preferred_redshift_scale(self):
        """
        Return a scaling factor to convert the redshift into the user's
        preferred units.
        """

        as_velocity = self._configuration.get("settings", {})\
            .get("redshift_as_velocity", False)
        return ("v", 299792.458, "km/s") if as_velocity else ("z", 1, "")


    def _check_forbidden_label_characters(self, characters):
        for parameter in self.grid_points.dtype.names:
            for character in characters:
                if character in parameter:
                    raise ValueError("parameter {0} contains the forbidden "\
                        "character '{1}'".format(parameter, character))


    def _latex_labels(self, labels):
        """
        LaTeX-ify labels based on information provided in the configuration.
        """

        config = self._configuration.get("latex_labels", {})
        return [config.get(label, label) for label in labels]

    @property
    def hash(self):
        """ Return a MD5 hash of the JSON-dumped model configuration. """ 
        return \
            md5(yaml.safe_dump(self._configuration).encode("utf-8")).hexdigest()


    def validate(self):
        """ Check that the model configuration is valid. """

        assert "model" in self._configuration



    @property
    def parameters(self):
        """ Return the model parameters. """

        try:
            return self._parameters

        except AttributeError:
            None

        parameters = []
        parameters.extend(self.grid_points.dtype.names)

        model_configuration = self._configuration.get("model", {})

        # Continuum
        # Note: continuum cannot just be 'True', it should be a dictionary
        n = len(parameters)
        continuum_config = model_configuration.get("continuum", False)
        if continuum_config is not False:
            if not isinstance(continuum_config, dict):
                raise TypeError("incorrect continuum type")
            for name in set(self.channel_names).intersection(continuum_config):
                degree = continuum_config[name]
                if degree is False:
                    continue

                # Note: N_coefficients = degree + 1
                # Example: degree = 3: a^3 + b^2 + c^1 + d 
                parameters.extend(["continuum_{0}_{1}".format(name, n) \
                    for n in range(degree + 1)])

        # Redshift
        parameters.extend(channel_parameters("z", self.channel_names,
            model_configuration.get("redshift", False)))

        # Spectral resolution
        resolution_parameters = channel_parameters(
            "resolution", self.channel_names,
            model_configuration.get("spectral_resolution", False))
        parameters.extend(resolution_parameters)

        # Underestimated variance
        parameters.extend(channel_parameters(
            "ln_f", self.channel_names,
            model_configuration.get("underestimated_variance", False)))

        # Outlier pixels
        if model_configuration.get("outlier_pixels", False):
            parameters.extend(["Po", "Vo"])

        assert len(parameters) == len(set(parameters)), "Duplicate parameters!"

        self._resolution_parameters = resolution_parameters
        self._parameters = tuple(parameters)

        return self._parameters


    def _overlapping_channels(self, wavelengths):
        """
        Return the channels that match the given wavelength array.
        """

        sizes = self.meta["channel_sizes"]
        min_a, max_a = wavelengths.min(), wavelengths.max()

        matched_channel_names = []
        for i, (name, size) in enumerate(zip(self.channel_names, sizes)):
            si = sum(sizes[:i])
            min_b, max_b = self.wavelengths[[si, si + size - 1]]
            if max_a > min_b and min_a < max_b:
                matched_channel_names.append(name)

        return matched_channel_names

    @property
    def _faux_data(self):
        data = []
        for i, px in enumerate(self.meta["channel_sizes"]):
            si = np.sum(self.meta["channel_sizes"][:i])
            data.append(specutils.Spectrum1D(disp=self.wavelengths[si:si+px],
                flux=np.ones(px)))
        return data


    def _format_data(self, data):
        """
        Sort the data in blue wavelengths to red, and ignore any spectra that
        have entirely non-finite or negative fluxes.
        """
        return [spectrum for spectrum in \
            sorted(data if isinstance(data, (list, tuple)) else [data],
                key=lambda x: x.disp[0]) if np.any(np.isfinite(spectrum.flux))]

    def _apply_data_mask(self, data):
        """
        Apply pre-defined masks to the data.
        """

        data = self._format_data(data)

        masked_data, pixels_affected = [], 0
        data_mask = self._configuration.get("masks", {}).get("data", [])
        for spectrum in data:
            masked_spectrum = spectrum.copy()
            for start, end in data_mask:
                idx = np.clip(
                    masked_spectrum.disp.searchsorted([start, end]) + [0, 1],
                    0, masked_spectrum.disp.size)
                masked_spectrum.flux[idx[0]:idx[1]] = np.nan
                pixels_affected += np.clip(np.ptp(idx) - 1, 0, None)
            masked_data.append(masked_spectrum)
        
        logger.debug("{0} observed pixels were masked according to the data "
            "mask: {1}".format(pixels_affected, data_mask))

        return (masked_data, pixels_affected)

    def _model_mask(self, wavelengths=None):
        """
        Apply pre-defined model masks.
        """

        if wavelengths is None:
            wavelengths = self.wavelengths
        wavelengths = np.array(wavelengths)

        mask = np.ones_like(wavelengths, dtype=bool)
        model_mask = self._configuration.get("masks", {}).get("model", [])
        logger.debug("Applying model mask: {0}".format(model_mask))
        for start, end in model_mask:
            idx = np.clip(wavelengths.searchsorted([start, end]) + [0, 1], 0,
                wavelengths.size)
            mask[idx[0]:idx[1]] = False
        return mask


    def _match_channels_to_data(self, data):
        """
        Match observed data to a channel, and return possibly superfluous model
        parameters.
        """

        data = self._format_data(data)

        matched_channels = []
        for spectrum in data:
            match = self._overlapping_channels(spectrum.disp)
            if not match:
                # TODO: make this warning only appear once.
                logger.warn("No model spectra found matching data channel from "
                    "{0:.0f}-{1:.0f} Angstroms. These data will be ignored."\
                    .format(spectrum.disp[0], spectrum.disp[-1]))

            if len(match) > 1:
                raise ValueError("cannot match multiple channels for one spec")

            if len(match) == 0:
                match = [None]

            matched_channels.append(match[0])

        missing_channels = set(self.channel_names).difference(matched_channels)

        # Now which parameters should be ignored?
        ignore_parameters = []
        if missing_channels:
            for channel in missing_channels:
                ignore_parameters.extend(
                    set(self.parameters).intersection([_.format(channel) \
                        for _ in ("resolution_{}", "f_{}", "z_{}")]))
                
                ignore_parameters.extend(["continuum_{0}_{1}".format(channel, i)\
                    for i in range(self._configuration["model"].get("continuum",
                        {}).get(channel, -1) + 1)])

        if ignore_parameters:
            logger.warn("Ignoring the following model parameters because there "
                "are no data for that channel: {}".format(", ".join(
                    ignore_parameters)))

        return (matched_channels, missing_channels, ignore_parameters)


    def _initial_proposal_distribution(self, parameters, theta, size,
        default_std=1e-4):
        """
        Generate an initial proposal distribution around the point theta.
        """

        missing_parameters = set(parameters).difference(theta)
        if missing_parameters:
            raise ValueError("cannot create initial proposal distribution "\
                "because the following parameters are missing: {}".format(
                    ", ".join(missing_parameters)))

        std = np.ones(len(parameters), dtype=float)
        
        initial_proposal_stds \
            = self._configuration.get("initial_proposal_stds", {})

        p0 = np.array([theta[p] for p in parameters])
        std = np.array(map(float, [initial_proposal_stds.get(p, default_std) \
            for p in parameters]))

        return np.vstack([p0 + std * np.random.normal(size=len(p0)) \
            for i in range(size)])


    def _chi_sq(self, theta, data, **kwargs):

        chi_sq, dof = 0, -1
        model_fluxes = self(theta, data, **kwargs)

        for spectrum, model_flux in zip(data, model_fluxes):
            chi_sqi = (spectrum.flux - model_flux)**2 / spectrum.variance
            finite = np.isfinite(chi_sqi)

            chi_sq += chi_sqi[finite].sum()
            dof += finite.sum()

        return (chi_sq, dof, model_fluxes)


    def cast(self, new_model_name, new_channels, output_dir=None, clobber=False,
        **kwargs):
        """
        Cast a new model from this model. The new model might have a restricted
        wavelength range, different channels, larger sampling/binning, and/or a
        lower spectral resolution.
        """

        output_dir = output_dir if output_dir is not None else os.getcwd()
        output_prefix = os.path.join(output_dir, new_model_name)

        # Validation.
        if not isinstance(new_channels, dict):
            raise TypeError("channels must be a dictionary object")

        for name, descr in new_channels.iteritems():
            if not isinstance(descr, (tuple, list, np.ndarray)) \
            or len(descr) != 2:
                raise TypeError("channel dictionary values must be a "\
                    "(wavelength_array, spectral_resolution)")

            new_wavelengths, spectral_resolution = descr
            if not isinstance(new_wavelengths, (list, np.ndarray)):
                raise TypeError("wavelengths for the new channel {} must be an "
                    "array".format(name))
            if 2 > len(new_wavelengths):
                raise ValueError("wavelength array for the new channel {0} only"
                    " contains {1} points".format(name, len(new_wavelengths)))
            if 0 > spectral_resolution:
                raise ValueError("spectral resolution for the new channel {} "
                    "must be a positive value".format(name))

        # Clobber.
        if not clobber:
            # Check to make sure the output files won't exist already.
            output_suffixes = (".yaml", ".pkl", "-wavelengths.memmap",
                "-intensities.memmap")
            for path in [output_prefix + suffix for suffix in output_suffixes]:
                if os.path.exists(path):
                    raise IOError("output filename {} already exists"\
                        .format(path))

        # Sort the proposed channel names from blue to red wavelengths.
        channel_names = []
        min_wavelengths = []
        for name, (wavelengths, resolution) in new_channels.iteritems():
            channel_names.append(name)
            min_wavelengths.append(np.min(wavelengths))

        channel_names = np.array(channel_names)[np.argsort(min_wavelengths)]

        # Get the wavelengths.
        channel_wavelengths = []
        wavelength_range = [self.wavelengths.min(), self.wavelengths.max()]
        for name in channel_names:

            new_wavelength = np.sort(new_channels[name][0])
            i = np.searchsorted(new_wavelength, wavelength_range)
            channel_wavelengths.append(new_wavelength[i[0]:i[1]])

            if i[0] != 0 or i[1] != len(new_wavelength):
                logger.warn("Clipping requested wavelength range in channel {0}"
                    " from {1:.0f}-{2:.0f} A to {3:.0f}-{4:.0f} A because the "
                    "region requested was not covered by the original model."\
                    .format(name, new_wavelength[0], new_wavelength[-1],
                        new_wavelength[i[0]], new_wavelength[i[1]]))

        channel_wavelengths = [new_channels[n][0] for n in channel_names]
        channel_sizes = [len(_) for _ in channel_wavelengths]
        num_pixels = sum(channel_sizes)

        # Find out which of the new channels overlap with the existing channels.
        channel_matches = {}
        for name, wavelengths in zip(channel_names, channel_wavelengths):

            overlapping_old_channels = self._overlapping_channels(wavelengths)

            # if there are no overlaps, do we have a problem?
            if len(overlapping_old_channels) == 0:
                raise ValueError("no channels overlapping")

            elif len(overlapping_old_channels) > 1:
                raise ValueError("too many channels overlapping!")

            channel_matches[name] = overlapping_old_channels[0]

        # Create the memory-mapped wavelength file.
        wavelengths_memmap = np.memmap(output_prefix + "-wavelengths.memmap",
            dtype="float32", mode="w+", shape=(num_pixels, ))
        wavelengths_memmap[:] = np.hstack(channel_wavelengths)
        wavelengths_memmap.flush()
        del wavelengths_memmap

        # Write the new configuration to file.
        with open(output_prefix + ".yaml", "w") as fp:
            header = "\n".join([
                "# Model created on {0} from previous model with hash {1}"\
                    .format(strftime("%Y-%m-%d %H:%M:%S"), self.hash),
                "# Grid parameters: {0}".format(
                    ", ".join(self.grid_points.dtype.names)),
                "# Channel names: {0}".format(
                    ", ".join(channel_names))
                ])
            configuration = self._configuration.copy()
            configuration.update({
                "model_grid": {
                    "grid_points": output_prefix + ".pkl",
                    "wavelengths": output_prefix + "-wavelengths.memmap",
                    "intensities": output_prefix + "-intensities.memmap"
                }
            })
            fp.write(header + "\n" + yaml.safe_dump(configuration, stream=None, 
                allow_unicode=True, default_flow_style=False))

        # Create the pickled model file, with meta data.
        meta = self.meta.copy()
        meta.update({
            "channel_names": channel_names,
            "channel_sizes": channel_sizes,
            "channel_resolutions": [new_channels[n][1] for n in channel_names]
        })
        with open(output_prefix + ".pkl", "wb") as fp:
            pickle.dump((self.grid_points, meta), fp, -1)
            
        # Create the rebinning matrices.
        convolutions = []
        wavelength_indices = []
        fast_binning \
            = self._configuration.get("settings", {}).get("fast_binning", False)
        for name in channel_names:
            new_wavelengths, spectral_resolution = new_channels[name]

            indices = np.clip(self.wavelengths.searchsorted(
                [new_wavelengths.min(), new_wavelengths.max()]) + [0, 1],
                0, self.wavelengths.size - 1)
            wavelength_indices.append(indices)
            old_wavelengths = self.wavelengths[indices[0]:indices[1]]
            logger.debug("Casting {0} channel from [{1:.0f}, {2:.0f}] to "\
                "[{3:.0f}, {4:.0f}]".format(name,
                    old_wavelengths[0], old_wavelengths[-1],
                    new_wavelengths[0], new_wavelengths[-1]))

            if fast_binning:
                if spectral_resolution is None \
                or not np.isfinite(spectral_resolution):
                    convolutions.append(lambda nw, ow, of: \
                        np.interp(nw, ow, of, left=np.nan, right=np.nan))

                else:
                    print("OK")
                    logger.debug("Using fast binning with spectral resolution")
                    R_scale = 2.3548200450309493 * new_wavelengths.mean()**2/ np.diff(new_wavelengths).mean()
                    convolutions.append(lambda nw, ow, of: \
                        np.interp(nw, ow,
                            gaussian_filter1d(of, R_scale/spectral_resolution**2),
                            left=np.nan, right=np.nan))

            else:
                if spectral_resolution is None \
                or not np.isfinite(spectral_resolution):
                    convolutions.append(lambda *_: _[2] * specutils.sample.resample(
                        old_wavelengths, new_wavelengths))
                else:
                    convolutions.append(lambda *_: _[2] * \
                        specutils.sample.resample_and_convolve(old_wavelengths,
                            new_wavelengths, new_resolution=spectral_resolution))
            
        # Load the intensities.
        intensities = np.memmap(
            self._configuration["model_grid"]["intensities"],
            shape=(self.grid_points.size, self.wavelengths.size),
            mode="r", dtype="float32")

        # Create a new intensities grid.
        cast_intensities = np.memmap(
            output_prefix + "-intensities.memmap",
            shape=(self.grid_points.size, num_pixels),
            mode="w+", dtype="float32")

        n = self.grid_points.size
        increment = int(n / 100)
        progressbar = kwargs.pop("__progressbar", False)
        if progressbar:
            print("Casting {} model:".format(new_model_name))

        for i in xrange(n):
            if progressbar:
                if (i % increment == 0):
                    sys.stdout.write("\r[{done}{not_done}] "
                        "{percent:3.0f}%".format(done="=" * int(i / increment),
                        not_done=" " * int((n - i)/ increment),
                        percent=100. * i/n))
                    sys.stdout.flush()
            else:
                logger.debug("Recasting point {0}/{1}".format(i, n))

            for j, (size, convolution, indices) \
            in enumerate(zip(channel_sizes, convolutions, wavelength_indices)):

                new_wavelengths, _ = new_channels[channel_names[j]]
                old_wl_idxs = np.clip(self.wavelengths.searchsorted(
                    [new_wavelengths.min(), new_wavelengths.max()]) + [0, 1],
                    0, self.wavelengths.size - 1)
                old_wavelengths = self.wavelengths[old_wl_idxs[0]:old_wl_idxs[1]]

                old_fluxes = np.copy(intensities[i, indices[0]:indices[1]])

                idx = sum(channel_sizes[:j])
                cast_intensities[i, idx:idx + size] = convolution(new_wavelengths, old_wavelengths, old_fluxes)

        """
        for j, (size, matrix, indices) \
        in enumerate(zip(channel_sizes, convolutions, wavelength_indices)):
            idx = sum(channel_sizes[:j])
            cast_intensities[:, idx:idx + size] = \
                intensities[:, indices[0]:indices[1]] * matrix
        """

        cast_intensities.flush()
        del cast_intensities
        if progressbar:
            print("\r")

        return True




def channel_parameters(parameter_prefix, channel_names, configuration_entry):
    """
    Return parameters specific to a channel for a model.
    """

    if configuration_entry in (True, False):
        return ([], [parameter_prefix])[configuration_entry]

    parameters = []
    if isinstance(configuration_entry, dict):
        for name in set(channel_names).intersection(configuration_entry):
            if configuration_entry[name] is True:
                parameters.append(parameter_prefix + "_" + name)
    else:
        raise TypeError("incorrect type")

    return parameters

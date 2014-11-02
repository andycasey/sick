# coding: utf-8

""" Validation funtions for the model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["validate"]

import logging

logger = logging.getLogger("sick")


def validate(configuration, channels, parameters):
    """
    Validate that the model has been specified properly.

    :returns:
        True
    """

    _validate_channels(configuration, channels, parameters)
    _validate_settings(configuration, channels, parameters)
    _validate_normalisation(configuration, channels, parameters)
    _validate_redshift(configuration, channels, parameters)
    _validate_convolve(configuration, channels, parameters)
    _validate_mask(configuration, channels, parameters)

    return True


def _validate_normalisation(configuration, channels, parameters):
    """
    Validate that the normalisation settings in the model are specified correctly.

    :returns:
        True if the normalisation settings for this model are specified correctly.

    :raises:
        KeyError if a model channel does not have a normalisation settings specified.
        TypeError if an incorrect data type is specified for a normalisation setting.
        ValueError if an incompatible data value is specified for a normalisation setting.
    """

    # Normalisation not required
    if not configuration["normalise"]:
        return True

    if not isinstance(configuration["normalise"], dict):
        raise TypeError("normalise setting must be boolean or a dictionary")

    for c, n_setup in configuration["normalise"].iteritems():
        if c not in channels:
            logger.warn("Ignoring non-existent channel {0} in normalisation"\
                " settings".format(c))
            continue

        if not isinstance(n_setup, dict):
            raise TypeError("per-channel normalisation setup must be a dictionary")

        if "order" not in n_setup.keys():
            raise KeyError("missing order setting for normalisation in "\
                "channel {0}".format(c))
    return True


def _validate_settings(configuration, channels, parameters):
    """
    Validate that the settings in the model are specified correctly.

    :returns:
        True if the settings for this model are specified correctly.

    :raises:
        KeyError if a model channel does not have a normalisation settings specified.
        TypeError if an incorrect data type is specified for a normalisation setting.
    """

    if "walkers" not in configuration["settings"].keys():
        logger.warn("Number of walkers not set (settings.walkers) in model"\
            " configuration file. Setting as {0}".format(2 * len(parameters)))
        configuration["settings"]["walkers"] = 2 * len(parameters)

    for key in ("burn", "sample", "walkers"):
        try: int(configuration["settings"][key])
        except (ValueError, TypeError) as e:
            raise TypeError("configuration setting settings.{0} must be"\
                " an integer-like type".format(key))

    if configuration["settings"]["walkers"] < 2*len(parameters):
        raise ValueError("number of walkers must be at least twice the "\
            "number of model parameters")

    if configuration["settings"]["burn"] > configuration["settings"]["sample"]:
        logger.warn("Number of burn-in steps exceeds the production quantity.")

    if "threads" in configuration["settings"] \
    and not isinstance(configuration["settings"]["threads"], (float, int)):
        raise TypeError("configuration setting 'settings.threads' must be an integer-like type")

    if configuration["settings"]["threads"] > configuration["settings"]["walkers"]:
        logger.warn("Number of threads exceeds the number of walkers.")
    return True


def _validate_redshift(configuration, channels, parameters):
    """
    Validate that the doppler shift settings in the model are specified correctly.

    :returns:
        True if the doppler settings for this model are specified correctly.
    """

    # Redshift and can be bool
    if isinstance(configuration["redshift"], bool):
        return True

    if not isinstance(configuration["redshift"], dict):
        raise TypeError("redshift must be boolean or a dictionary")

    for c, v in configuration["redshift"].iteritems():
        if c not in channels:
            logger.warn("Ignoring non-existent channel {0} in redshift "\
                "settings".format(c))
            continue
        if not isinstance(v, bool):
            raise TypeError("per-channel redshift must be boolean")
    return True


def _validate_convolve(configuration, channels, parameters):
    """
    Validate that the smoothing settings in the model are specified correctly.

    :returns:
        True if the smoothing settings for this model are specified correctly.
    """ 

    # Convolution not required
    if configuration["convolve"] == False:
        return True

    if not isinstance(configuration["convolve"], (dict, bool)):
        raise TypeError("convolve must be boolean or a dictionary")

    if isinstance(configuration["convolve"], dict):
        for c, v in configuration["convolve"].iteritems():
            if c not in channels:
                logger.warn("Ignoring non-existent channel {0} in convolve "\
                    "settings".format(c))
                continue
            if not isinstance(v, bool):
                raise TypeError("per-channel convolve must be boolean")
    return True


def _validate_channels(configuration, channels, parameters):
    """
    Validate that the channels in the model are specified correctly.

    :returns:
        True if the channels in the model are specified correctly.

    :raises:
        KeyError if no channels are specified.
        ValueError if an illegal character is present in any of the channel names.
    """

    key = ["channels", "cached_channels"]["cached_channels" in configuration.keys()]
    if key not in configuration:
        raise KeyError("no channels found in model file")

    for channel in channels:
        if "." in channel:
            raise ValueError("channel name '{0}' cannot contain a full-stop"\
                " character".format(channel))
    return True


def _validate_mask(configuration, channels, parameters):
    """
    Validate that the masks in the model are specified correctly.

    :returns:
        True if the masks in the model are specified correctly.

    :raises:
        TypeError if the masks are not specified correctly.
    """

    # Masks are optional
    if "mask" not in configuration.keys() \
    or configuration["mask"] is None:
        return True

    for region in configuration["mask"]:
        assert len(region) == 2, "Masks must be a list of regions (e.g. [start,"\
            " end])"
        if not isinstance(region[0], (int, float)) \
        or not isinstance(region[1], (int, float)):
            raise TypeError("masks must be a float-type")

    return True


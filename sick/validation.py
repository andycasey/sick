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

    _validate_channels(configuration, channels)
    _validate_settings(configuration, parameters)
    _validate_normalisation(configuration, channels)
    _validate_redshift(configuration, channels)
    _validate_convolve(configuration, channels)
    _validate_mask(configuration)

    return True


def _validate_normalisation(configuration, channels):
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
    if "normalise" not in configuration \
    or configuration["normalise"] == False:
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

        if isinstance(n_setup["order"], bool):
            raise TypeError("order for {} channel must be a positive integer".format(c))

        if (n_setup["order"] % 1) > 0:
            logger.warn("Setting order for {0} channel as integer ({1}, from {2})"\
                .format(c, int(n_setup["order"]), n_setup["order"]))
        try:
            configuration["normalise"][c]["order"] = int(n_setup["order"])
        except (TypeError, ValueError):
            raise TypeError("order for {} channel must be a positive integer".format(c))

        if n_setup["order"] < 0:
            raise ValueError("order for {} channel must be a positive integer".format(c))
    return True


def _validate_settings(configuration, parameters):
    """
    Validate that the settings in the model are specified correctly.

    :returns:
        True if the settings for this model are specified correctly.

    :raises:
        KeyError if a model channel does not have a normalisation settings specified.
        TypeError if an incorrect data type is specified for a normalisation setting.
    """

    if "settings" not in configuration:
        configuration["settings"] = {}

    if not isinstance(configuration["settings"], dict):
        raise TypeError("settings must be a dictionary")

    if "walkers" not in configuration["settings"].keys():
        logger.warn("Number of walkers not set (settings.walkers) in model"\
            " configuration file. Setting as {0}".format(2 * len(parameters)))
        configuration["settings"]["walkers"] = 2 * len(parameters)

    for key in ("burn", "sample", "walkers"):
        value = configuration["settings"][key]
        if isinstance(value, bool):
            raise TypeError("configuration setting settings.{} must be an "\
                "integer-like type".format(key))
        if not isinstance(value, int):
            raise TypeError("configuration setting settings.{0} must be"\
                " an integer-like type".format(key))
        if 0 >= value:
            raise ValueError("configuration setting settings.{} must be a "\
                "positive integer-like type".format(key))

    if (configuration["settings"]["walkers"] % 2) > 0:
        raise ValueError("number of walkers must be an even number")

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


def _validate_redshift(configuration, channels):
    """
    Validate that the doppler shift settings in the model are specified correctly.

    :returns:
        True if the doppler settings for this model are specified correctly.
    """

    if not "redshift" in configuration:
        return True

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


def _validate_convolve(configuration, channels):
    """
    Validate that the smoothing settings in the model are specified correctly.

    :returns:
        True if the smoothing settings for this model are specified correctly.
    """ 


    # Convolution not required
    if "convolve" not in configuration:
        return True

    # Can be boolean
    if isinstance(configuration["convolve"], bool):
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


def _validate_channels(configuration, channels):
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
        if not isinstance(channel, (str, unicode)):
            raise TypeError("channel must be a string-like object")

        if "." in channel:
            raise ValueError("channel name '{0}' cannot contain a full-stop"\
                " character".format(channel))
    return True


def _validate_mask(configuration):
    """
    Validate that the masks in the model are specified correctly.

    :returns:
        True if the masks in the model are specified correctly.

    :raises:
        TypeError if the masks are not specified correctly.
    """

    
    # Masks are optional
    if "mask" not in configuration.keys() \
    or configuration["mask"] == False \
    or configuration["mask"] is None:
        return True

    if isinstance(configuration["mask"], dict):
        raise TypeError("mask must be a list-type of regions")

    for region in configuration["mask"]:
        if len(region) != 2:
            raise TypeError("Masks must be a list of regions (e.g. [start,"\
                " end])")
        
        if not isinstance(region[0], (int, float)) \
        or not isinstance(region[1], (int, float)):
            raise TypeError("masks must be a float-type")

        if region[0] > region[1]:
            logger.warn("Masked region [{0}, {1}] has the start region bigger"\
                " than the end region: {0} > {1}".format(*region))
        elif region[0] == region[1]:
            logger.warn("Masked regions {0} and {1} are the same point".format(
                *region))

    return True


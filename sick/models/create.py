#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Create models for *sick* """

from __future__ import division, print_function

__all__ = ("create", )
__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import os
import yaml

import numpy as np
from astropy.io import fits
from astropy.table import Table

from time import strftime

logger = logging.getLogger("sick")


def load_simple_data(filename, **kwargs):
    # parse a txt/fits file with ease.

    logger.debug("Opening {}".format(filename))
    fits_extensions = (".fit", ".fits", ".fit.gz", ".fits.gz")
    if any(map(lambda _: filename.endswith(_), fits_extensions)):
        # laod as fits.
        with fits.open(filename) as image:
            extension_index = kwargs.pop("extension", None)
            if extension_index is None:
                # Get first extension with data.
                for extension_index, extension in enumerate(image):
                    if extension.data is not None: break
                else:
                    raise IOError("no valid data in {}".format(filename))

            data = image[extension_index].data
        return data

    else:
        return np.loadtxt(filename, **kwargs)


def create(output_prefix, grid_flux_filename, wavelength_filenames,
    clobber=False, grid_flux_filename_format="csv", **kwargs):
    """
    Create a new *sick* model from files describing the parameter names, fluxes,
    and wavelengths.
    """

    if not clobber:
        # Check to make sure the output files won't exist already.
        output_suffixes = (".yaml", ".pkl", "-wavelengths.memmap",
            "-intensities.memmap")
        for path in [output_prefix + suffix for suffix in output_suffixes]:
            if os.path.exists(path):
                raise IOError("output filename {} already exists".format(path))

    # Read the grid_flux filename.
    # param1 param2 param3 param4 channelname1 channelname2
    kwds = kwargs.pop("__grid_flux_filename_kwargs", {})
    kwds.update({"format": grid_flux_filename_format})
    grid_flux_tbl = Table.read(grid_flux_filename, **kwds)

    # Distinguish column names between parameters (real numbers) and filenames
    str_columns = \
        np.array([_[1].startswith("|S") for _ in grid_flux_tbl.dtype.descr])

    # Check the number of channels provided.
    if str_columns.sum() != len(wavelength_filenames):
        raise ValueError("expected {0} wavelength filenames because {1} has {0}"
            " string columns ({2}) but found {3} wavelength filenames".format(
                sum(str_columns), grid_flux_filename, 
                ", ".join(np.array(grid_flux_tbl.colnames)[str_columns]), 
                len(wavelength_filenames)))

    # Create a record array of the grid points.
    grid_points = \
        grid_flux_tbl.as_array()[np.array(grid_flux_tbl.colnames)[~str_columns]]

    # To-do: make sure they are all floats.

    # Sort the grid points.
    grid_indices = grid_points.argsort(order=grid_points.dtype.names)
    grid_points = grid_points[grid_indices]
    grid_flux_tbl = grid_flux_tbl[grid_indices]

    # Check the wavelength filenames.
    channel_wavelengths = np.array(map(load_simple_data, wavelength_filenames))

    # Sort the channels by starting wavelength.
    c_indices = np.argsort([each.min() for each in channel_wavelengths])
    channel_names = np.array(grid_flux_tbl.colnames)[str_columns][c_indices]
    channel_wavelengths = channel_wavelengths[c_indices]
    channel_sizes = [len(_) for _ in channel_wavelengths]
    num_pixels = sum(channel_sizes)

    # Create the model YAML file.
    with open(output_prefix + ".yaml", "w") as fp:
        header = "\n".join([
            "# Model created on {0}".format(strftime("%Y-%m-%d %H:%M:%S")),
            "# Grid parameters: {0}".format(", ".join(grid_points.dtype.names)),
            "# Channel names: {0}".format(", ".join(channel_names))
            ])
        fp.write(header + "\n" + yaml.safe_dump({ "model_grid": {
                "grid_points": output_prefix + ".pkl",
                "intensities": output_prefix + "-intensities.memmap",
                "wavelengths": output_prefix + "-wavelengths.memmap"
            }}, stream=None, allow_unicode=True, default_flow_style=False))

    # Create the pickled model file, with meta data.
    metadata = {
        "grid_flux_filename": grid_flux_filename,
        "wavelength_filenames": wavelength_filenames,
        "channel_names": channel_names,
        "channel_sizes": channel_sizes,
        "sick_version": None
    }
    logger.debug("Dumping grid points and metadata to file")
    with open(output_prefix + ".pkl", "wb") as fp:
        pickle.dump((grid_points, metadata), fp, -1)

    # Create the memory-mapped dispersion file.
    logger.debug("Creating memory-mapped dispersion file.")
    wavelengths_memmap = np.memmap(output_prefix + "-wavelengths.memmap",
        dtype="float32", mode="w+", shape=(num_pixels, ))
    wavelengths_memmap[:] = np.hstack(channel_wavelengths)
    wavelengths_memmap.flush()
    del wavelengths_memmap

    # Create the memory-mapped intensities file.
    logger.debug("Creating memory-mapped intensities file.")
    intensities_memmap = np.memmap(output_prefix + "-intensities.memmap",
        shape=(grid_points.size, num_pixels), dtype="float32",
        mode="w+")
    intensities_memmap.flush()

    n = len(grid_flux_tbl)
    for i, row in enumerate(grid_flux_tbl):
        logger.debug("Loading point {0}/{1} into the intensities map"\
            .format(i + 1, n))
        j = 0
        for channel_name in channel_names:
            try:
                data = load_simple_data(row[channel_name])
            except:
                logger.exception("Could not load data from {0} for channel {1}"\
                    .format(row[channel_name], channel_name))
                raise
            intensities_memmap[i, j:j + data.size] = data
            j += data.size

    intensities_memmap.flush()
    del intensities_memmap

    return True

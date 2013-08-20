# coding: utf-8

""" General utilities for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os

# Third-party
import numpy as np

__all__ = ['find_spectral_overlap']

def find_spectral_overlap(dispersion_maps, interval_resolution=1):
    """Checks whether the dispersion maps overlap or not.

    Inputs
    ------
    dispersion_maps : list of list-types of length 2+
        The dispersion maps in the format [(wl_1, wl_2, ... wl_N), ..., (wl_start, wl_end)]
    
    interval_resolution : float, Angstroms
        The resolution at which to search for overlap. Any overlap less than the
        `interval_resolution` may not be detected.

    Returns
    -------
    None if no overlap is found, otherwise the wavelength near the overlap is returned.
    """

    all_min = map(np.min, dispersion_maps)
    all_max = map(np.max, dispersion_maps)

    interval_tree_disp = np.arange(all_min, all_max + interval_tree_resolution, interval_tree_resolution)
    interval_tree_flux = np.zeros(len(interval_tree_disp))

    for dispersion_map in dispersion_maps:

        wlstart, wlend = np.min(dispersion_map), np.max(dispersion_map)
        idx = np.searchsorted(interval_tree_disp, [wlstart, wlend + interval_tree_resolution])

        interval_tree_flux[idx[0]:idx[1]] += 1

    # Any overlap?
    if np.max(interval_tree_flux) > 1:
        idx = np.where(interval_tree_flux > 1)[0]
        wavelength = interval_tree_disp[idx[0]]
        return wavelength
    
    else:
        return None



def map_apertures(observed_dispersions, model_dispersions):
    """References model spectra to each observed spectra based on their wavelengths.

    Inputs
    ------
    observed_dispsersions : a list of observed dispersion maps (e.g. list-types full of floats)
        The dispersion maps for all observed apertures.

    model_dispersions : a list of model dispersion maps (e.g. list-types full of floats)
        The dispersion maps for all model apertures.
    """

    mapped_apertures = {}
    for i, observed_dispersion in enumerate(observed_dispersions):

        # Initialise the list
        mapped_apertures[i] = []

        observed_wlmin = np.min(observed_dispersion)
        observed_wlmax = np.max(observed_dispersion)

        for j, model_dispersion in enumerate(model_dispersions):

            model_wlmin = np.min(model_dispersion)
            model_wlmax = np.max(model_dispersion)

            # Is there overlap?
            if (model_wlmin < observed_wlmin and observed_wlmax < model_wlmax) \
            or (observed_wlmin < model_wlmin and model_wlmax < observed_wlmax) \
            or (model_wlmin < observed_wlmin and (observed_wlmin < model_wlmax and model_wlmax < observed_wlmax)) \
            or ((observed_wlmin < model_wlmin and model_wlmin < observed_wlmax) and observed_wlmax < model_wlmax):
                mapped_apertures[i] += [j]

    return mapped_apertures





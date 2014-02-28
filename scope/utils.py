# coding: utf-8

""" General utilities for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np

__all__ = ["human_readable_digit", "find_spectral_overlap"]


def human_readable_digit(number):
    millnames = ["", "thousand", "million", "billion", "trillion"]
    millidx = max(0, min(len(millnames)-1,
                      int(np.floor(np.log10(abs(number))/3.0))))
    return "{digit:.1f} {multiple}".format(
        digit=number/10**(3*millidx),
        multiple=millnames[millidx])


def find_spectral_overlap(dispersion_maps, interval_resolution=1):
    """ Checks whether the dispersion maps overlap or not.

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

    all_min = np.min(map(np.min, dispersion_maps))
    all_max = np.max(map(np.max, dispersion_maps))

    interval_tree_disp = np.arange(all_min, 
        all_max + interval_resolution, interval_resolution)
    interval_tree_flux = np.zeros(len(interval_tree_disp))

    for dispersion_map in dispersion_maps:

        wlstart, wlend = np.min(dispersion_map), np.max(dispersion_map)
        idx = np.searchsorted(interval_tree_disp, 
            [wlstart, wlend + interval_resolution])
        interval_tree_flux[idx[0]:idx[1]] += 1

    # Any overlap?
    if np.max(interval_tree_flux) > 1:
        idx = np.where(interval_tree_flux > 1)[0]
        wavelength = interval_tree_disp[idx[0]]
        return wavelength
    
    else:
        return None
# coding: utf-8

""" General utilities for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["human_readable_digit", "find_spectral_overlap", "latexify",
    "unique_preserved_list"]

import numpy as np

def latexify(labels, overwrite_common_labels=None):

    common_labels = {
        "teff": "$T_{\\rm eff}$ [K]",
        "feh": "[Fe/H]",
        "logg": "$\log{g}$",
        "alpha": "[$\\alpha$/Fe]",
        "jitter": "$lnf$"
    }

    if overwrite_common_labels is not None:
        common_labels.update(overwrite_common_labels)

    latex_labels = []
    for label in labels:

        if label in common_labels.keys():
            latex_labels.append(common_labels[label])
        
        elif label.startswith("jitter."):
            aperture = label.split(".")[1]
            latex_labels.append("$ln(f_{{{0}}})$".format(aperture))

        elif label.startswith("doppler_shift."):
            aperture = label.split(".")[1]
            latex_labels.append("$V_{{los,{{{0}}}}}$ [km/s]".format(aperture))

        elif label.startswith("smooth_model_flux."):
            aperture = label.split(".")[1]
            latex_labels.append("$\sigma_{{{0}}}$ [$\AA$]".format(aperture))

        elif label.startswith("normalise_observed."):
            aperture, coefficient = label.split(".")[1:]
            if coefficient == "s":
                latex_labels.append("$s_{{{0}}}$".format(aperture))
            else:
                latex_labels.append("${0}_{1}$".format(aperture[0], coefficient[1:]))

        else:
            latex_labels.append(label)

    return latex_labels


def unique_preserved_list(original_list):
    seen = set()
    seen_add = seen.add
    return [x for x in original_list if x not in seen and not seen_add(x)]


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

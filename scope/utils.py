# coding: utf-8

""" General utility functions  """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["human_readable_digit", "find_spectral_overlap", "latexify",
    "unique_preserved_list"]

import numpy as np

def latexify(labels, overwrite_common_labels=None):
    """
    Return LaTeX-ified labels.

    Args:
        labels (list of str objects): List of strings to LaTeX-ify.

        overwrite_common_labels (dict): Dictionary of common labels to use.

    Returns:
        List of LaTeX labels.
    """

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

        elif label.startswith("convolve."):
            aperture = label.split(".")[1]
            latex_labels.append("$\sigma_{{{0}}}$ [$\AA$]".format(aperture))

        elif label.startswith("normalise_observed."):
            aperture, coefficient = label.split(".")[1:]
            latex_labels.append("${0}_{1}$".format(aperture[0], coefficient))

        else:
            latex_labels.append(label)

    return latex_labels


def unique_preserved_list(original_list):
    """
    Return the unique items of a list in their original order.
    """
    
    seen = set()
    seen_add = seen.add
    return [x for x in original_list if x not in seen and not seen_add(x)]


def human_readable_digit(number):
    """
    Return a more human-readable (English) digit.
    """

    millnames = ["", "thousand", "million", "billion", "trillion"]
    millidx = max(0, min(len(millnames)-1, int(np.floor(np.log10(abs(number))/3.0))))
    return "{0:.1f} {1}".format(number/10**(3*millidx), millnames[millidx])


def find_spectral_overlap(dispersion_maps, interval_resolution=1):
    """ 
    Finds spectral overlap between dispersion maps.

    Args:
        dispersion_maps (list of np.arrays): Dispersion maps.

        interval_resolution (float): The interval to check for overlap within.
    
    Returns:
        None if no spectral overlap is found, otherwise it returns the approximate
        wavelength where there is spectral overlap.
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

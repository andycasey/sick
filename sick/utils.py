# coding: utf-8

""" General utility functions  """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["human_readable_digit", "latexify", "unique_preserved_list", "wrapper"]

import numpy as np

class wrapper(object):
    """ A general wrapper to pickle functions. """

    def __init__(self, func, args):
        self.func = func
        self.args = args
        
    def __call__(self, x):
        return self.func(x, *self.args)


def latexify(labels, default_latex_labels=None):
    """
    Return a LaTeX-ified label.

    :param labels:
        The label(s) to latexify. 

    :type labels:
        str or iterable of str objects

    :param default_latex_labels: [optional]
        Dictionary of default labels to use.

    :type default_latex_labels:
        dict

    :returns:
        LaTeX-ified label.

    :rtype:
        str or iterable of str objects
    """

    common_labels = {
        "teff": "$T_{\\rm eff}$ [K]",
        "feh": "[Fe/H]",
        "logg": "$\log{g}$",
        "alpha": "[$\\alpha$/Fe]"
    }

    if default_latex_labels is not None:
        common_labels.update(default_latex_labels)
    
    listify = True
    if isinstance(labels, str):
        listify = False
        labels = [labels]

    latex_labels = []
    for label in labels:

        if label in common_labels.keys():
            latex_labels.append(common_labels[label])
        
        elif label.startswith("f."):
            aperture = label.split(".")[1]
            latex_labels.append("$ln(f_{{{0}}})$".format(aperture))

        elif label.startswith("v."):
            aperture = label.split(".")[1]
            latex_labels.append("$V_{{rad,{{{0}}}}}$ [km/s]".format(aperture))

        elif label.startswith("z."):
            aperture = label.split(".")[1]
            latex_labels.append("$z_{{{0}}}$".format(aperture))

        elif label.startswith("convolve."):
            aperture = label.split(".")[1]
            latex_labels.append("$\sigma_{{{0}}}$ [$\AA$]".format(aperture))

        elif label.startswith("normalise."):
            aperture, coefficient = label.split(".")[1:]
            latex_labels.append("${0}_{{{1}}}$".format(aperture[0], coefficient))

        else:
            latex_labels.append(label)

    if not listify:
        return latex_labels[0]

    return latex_labels


def unique_preserved_list(original_list):
    """ Return the unique items of a list in their original order. """
    
    seen = set()
    seen_add = seen.add
    return [x for x in original_list if x not in seen and not seen_add(x)]


def human_readable_digit(number):
    """ Return a digit in a human-readable string form. """

    if 0 >= number:
        return "{0:.1f} ".format(number)

    millnames = ["", "thousand", "million", "billion", "trillion"]
    millidx = max(0, min(len(millnames)-1, int(np.floor(np.log10(abs(number))/3.0))))
    return "{0:.1f} {1}".format(number/10**(3*millidx), millnames[millidx])
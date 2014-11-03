# coding: utf-8

""" General utility functions for sick """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["default_output_prefix", "human_readable_digit", "latexify", 
    "sample_ball", "unique_preserved_list", "update_recursively", "wrapper"]

import collections
import numpy as np
import os

from emcee.utils import sample_ball

def update_recursively(original, new):
    """
    Recursively update a nested dictionary.
    
    :param original:
        The original nested dictionary to update.

    :type original:
        dict

    :param new:
        The nested dictionary to use to update the original.

    :type new:
        dict

    :returns:
        The updated original dictionary.

    :rtype:
        dict
    """

    for k, v in new.iteritems():
        if isinstance(v, collections.Mapping) \
        and isinstance(original.get(k, None), collections.Mapping):
            r = update_recursively(original.get(k, {}), v)
            original[k] = r
        else:
            original[k] = new[k]
    return original


def default_output_prefix(filenames):
    """
    Return a default filename prefix for output files based on the input files.

    :param filenames:
        The input filename(s):

    :type filenames:
        str or list of str

    :returns:
        The extensionless common prefix of the input filenames:

    :rtype:
        str
    """

    if isinstance(filenames, (str, )):
        filenames = [filenames]
    common_prefix, ext = os.path.splitext(os.path.commonprefix(
        map(os.path.basename, filenames)))
    return common_prefix if len(common_prefix) > 0 else "sick"


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

    # Check out dat stellar bias.
    common_labels = {
        "teff": "$T_{\\rm eff}$ [K]",
        "feh": "[Fe/H]",
        "logg": "$\log{g}$",
        "alpha": "[$\\alpha$/Fe]"
    }

    # Don't worry; I got yo covered.
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
            aperture, coeff = label.split(".")[1:]
            latex_labels.append("${0}_{{{1}}}$".format(aperture[0], coeff[1:]))

        else:
            latex_labels.append(label)

    if not listify:
        return latex_labels[0]

    return latex_labels


def unique_preserved_list(original_list):
    """
    Return the unique items of a list in their original order.

    :param original_list:
        A list of items that may have duplicate entries.

    :type original_list:
        list

    :returns:
        A list with unique entries with the original order preserved.

    :rtype:
        list
    """
    
    seen = set()
    seen_add = seen.add
    return [x for x in original_list if x not in seen and not seen_add(x)]


def human_readable_digit(number):
    """
    Return a digit in a human-readable string form.

    :param number:
        The obfuscated number.

    :type number:
        float

    :returns:
        A more human-readable version of the input number.

    :rtype:
        str
    """

    if 0 >= number:
        return "{0:.1f} ".format(number)

    word = ["", "thousand", "million", "billion", "trillion"]
    millidx = max(0, min(len(word)-1, int(np.floor(np.log10(abs(number))/3.0))))
    return "{0:.1f} {1}".format(number/10**(3*millidx), word[millidx])


class wrapper(object):
    """ A general wrapper to pickle functions. """

    def __init__(self, func, args):
        self.func = func
        self.args = args
        
    def __call__(self, x):
        return self.func(x, *self.args)
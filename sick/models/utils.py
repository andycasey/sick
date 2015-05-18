#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" General utility functions """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ("human_readable_digit", "unique_preserved_list", "update_recursively")

import collections
import numpy as np

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


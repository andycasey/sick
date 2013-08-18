# coding: utf-8

""" Handles the loading and interpolation of flux models for SCOPE. """

from __future__ import division, print_function

__author__ = "Andy Casey <acasey@mso.anu.edu.au>"

# Standard library
import logging
import os
import re

from glob import glob

# Third-party
import numpy as np
import pyfits


class Models(object):
    """Class for interpolating model fluxes for SCOPE."""

    def __init__(self, configuration):
        self.configuration = configuration

        # Dispersions
        self.dispersion = {}
        for beam, dispersion_filename in configuration['models']['dispersion_filenames'].iteritems():
            self.dispersion[beam] = load_model_data(dispersion_filename)

        grid_points = {}
        flux_filenames = {}

        # Read the points from filenames
        for beam in configuration['models']['flux_filenames']:
            folder = configuration['models']['flux_filenames'][beam]['folder']
            re_match = configuration['models']['flux_filenames'][beam]['re_match']

            all_filenames = glob(os.path.join(folder, '*'))

            points = []
            matched_filenames = []
            for filename in all_filenames:
                match = re.match(re_match, os.path.basename(filename))

                if match is not None:
                    if not hasattr(self, 'point_dtypes'):
                        point_dtypes = []
                        groups = match.groups()

                        groupdict = match.groupdict()
                        for value in match.groupdict().itervalues():
                            if groups.count(value) > 1: break
                            point_dtypes.append(match.groupdict().keys()[groups.index(value)])

                        if len(point_dtypes) == len(groups):
                            self.point_dtypes = point_dtypes

                    points.append(map(float, match.groups()))
                    matched_filenames.append(filename)

            grid_points[beam] = points
            flux_filenames[beam] = matched_filenames

        # If it's just the one beam, it's easy!
        first_beam = configuration['models']['flux_filenames'].keys()[0]
        self.grid_points = np.array(grid_points[first_beam])
        
        if len(configuration['models']['flux_filenames'].keys()) == 1:
            self.flux_filenames = flux_filenames[first_beam]

            return None

        else:
            self.flux_filenames = {first_beam: flux_filenames[first_beam]}

        # Put all points and filenames on the one scale
        for beam in configuration['models']['flux_filenames'].keys()[1:]:
            
            points = grid_points[beam]
            if len(points) != len(self.grid_points):
                raise ValueError("number of model points found in {first_beam} beam ({num_first_beam})"
                    " did not match the number in {this_beam} beam ({num_this_beam})"
                    .format(first_beam=first_beam, num_first_beam=len(self.grid_points), this_beam=beam,
                        num_this_beam=len(points)))

            sort_indices = []

            for point in points:
                index = np.where(np.all(np.equal(self.grid_points - point, np.zeros(len(point))), 1))[0][0]
                sort_indices.append(index)

            self.flux_filenames[beam] = [flux_filenames[beam][index] for index in sort_indices]

        return None






def load_model_data(filename, **kwargs):
    """Loads dispersion/flux values from a given filename. This can be either a 1-column ASCII
    file, or a single extension FITS file.

    Inputs
    ------
    filename : `str`
        The filename to load the values from.
    """

    if not os.path.exists(filename):
        raise IOError("filename '{filename}' does not exist".format(filename=filename))

    try:
        image = pyfits.open(filename, **kwargs)

    except:
        data = np.loadtxt(filename, **kwargs)

    else:
        data = image[0].data
        image.close()

    finally:
        return data



        
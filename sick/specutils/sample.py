#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

""" Resampling and convolution functionality for spectra. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ("resample", "resample_and_convolve")

import logging
import numpy as np
from scipy import sparse

from sick.utils import lru_cache

logger = logging.getLogger("sick")

LRU_SIZE = 25

def resample_and_convolve(old_wavelengths, new_wavelengths, new_resolution,
    old_resolution=np.inf, threshold=5):

    N, M = (new_wavelengths.size, old_wavelengths.size)
    threshold_pixels = threshold/np.diff(old_wavelengths).max()

    # Calculate the width of the kernel at each point.
    # [TODO] should this actually be squared???
    fwhms = (new_wavelengths/new_resolution)**2
    if np.isfinite(old_resolution):
        assert old_resolution > new_resolution
        fwhms -= (new_wavelengths/old_resolution)**2

    # 2.355 ~= 2 * sqrt(2*log(2))
    sigmas = fwhms/2.3548200450309493
    N_kernel_pixels = int(np.ceil(sigmas[-1] * threshold_pixels))
    integer_offsets = old_wavelengths.searchsorted(new_wavelengths)

    # For +/- N_kernel_pixels at each point, calculate the kernel and retain
    # the indices.
    ios = integer_offsets[0]
    indices = np.tile(np.arange(ios - N_kernel_pixels,
        ios + N_kernel_pixels), N).reshape(N, 2 * N_kernel_pixels)
    indices[1:, :] += np.cumsum(np.diff(integer_offsets)).reshape(-1, 1)
    indices = np.clip(indices, 0, M - 1)

    x_indices = indices.flatten()
    y_indices = np.repeat(np.arange(N), 2 * N_kernel_pixels)

    # Calculate the kernel at each point.
    pdf = np.exp(
        -(old_wavelengths[indices].T - new_wavelengths)**2/(2*sigmas**2))
    pdf /= pdf.sum(axis=0)

    return sparse.coo_matrix((pdf.T.flatten(), (x_indices, y_indices)),
        shape=(M, N))


def resample(old_wavelengths, new_wavelengths):
    """
    Resample a spectrum to a new wavelengths map while conserving total flux.

    :param old_wavelengths:
        The original wavelengths array.

    :type old_wavelengths:
        :class:`numpy.array`

    :param new_wavelengths:
        The new wavelengths array to resample onto.

    :type new_wavelengths:
        :class:`numpy.array`
    """

    data = []
    old_px_indices = []
    new_px_indices = []
    for i, new_wl_i in enumerate(new_wavelengths):

        # These indices should span just over the new wavelength pixel.
        indices = np.unique(np.clip(
            old_wavelengths.searchsorted(new_wavelengths[i:i + 2], side="left")\
                + [-1, +1], 0, old_wavelengths.size - 1))
        N = np.ptp(indices)

        if N == 0:
            # 'Fake' pixel.
            data.append(np.nan)
            new_px_indices.append(i)
            old_px_indices.extend(indices)
            continue

        # Sanity checks.
        assert (old_wavelengths[indices[0]] <= new_wl_i \
            or indices[0] == 0)
        assert (new_wl_i <= old_wavelengths[indices[1]] \
            or indices[1] == old_wavelengths.size - 1)

        fractions = np.ones(N)

        # Edges are handled as fractions between rebinned pixels.
        _ = np.clip(i + 1, 0, new_wavelengths.size - 1)
        lhs = old_wavelengths[indices[0]:indices[0] + 2]
        rhs = old_wavelengths[indices[-1] - 1:indices[-1] + 1]
        fractions[0]  = (lhs[1] - new_wavelengths[i])/np.ptp(lhs)
        fractions[-1] = (new_wavelengths[_] - rhs[0])/np.ptp(rhs)

        # Being binned to a single pixel. Prevent overflow from fringe cases.
        fractions = np.clip(fractions, 0, 1)
        fractions /= fractions.sum()

        data.extend(fractions) 
        new_px_indices.extend([i] * N) # Mark the new pixel indices affected.
        old_px_indices.extend(np.arange(*indices)) # And the old pixel indices.

    return sparse.csc_matrix((data, (old_px_indices, new_px_indices)),
        shape=(old_wavelengths.size, new_wavelengths.size))


def _fast_resample(old_wavelengths, new_wavelengths):

    # Map which pixels go to which index, assuming that the from_wavelengths
    # scale is linear.

    N, M = (new_wavelengths.size, old_wavelengths.size)
    pixel_edges = np.hstack([
        new_wavelengths[0] - (new_wavelengths[1] - new_wavelengths[0])/2.,
        new_wavelengths[:-1] + np.diff(new_wavelengths)/2.,
        new_wavelengths[-1] + (new_wavelengths[-1] - new_wavelengths[-2])/2.])

    fractional_indices_scale = (pixel_edges - old_wavelengths[0]) \
        * M/np.ptp(old_wavelengths)
    int_fractional_indices_scale = fractional_indices_scale.astype(int)

    upper_diagonals = 1 - fractional_indices_scale[:-1] % 1
    lower_diagonals = fractional_indices_scale[1:] % 1
    central_diagonal_pixel_width = np.clip(
        np.diff(int_fractional_indices_scale) - 1, 0, None)

    normalised_scale = 1.0/(upper_diagonals + central_diagonal_pixel_width \
        + lower_diagonals)

    # If the to_wavelength is linear then the normalised scale per pixel
    # would be the same for every pixel. Let's not assume that..
    px_normalised_scale = np.repeat(normalised_scale,
        central_diagonal_pixel_width)

    values = np.hstack([
        upper_diagonals * normalised_scale,
        lower_diagonals * normalised_scale,
        np.ones(central_diagonal_pixel_width.sum()) * px_normalised_scale])

    _ = np.arange(N)
    y_indices = \
        np.clip(np.hstack([_, _, np.repeat(_, central_diagonal_pixel_width)]),
            0, N - 1)
    x_indices = np.clip(np.hstack([
        int_fractional_indices_scale[:-1],
        int_fractional_indices_scale[1:],
        1 + np.repeat(int_fractional_indices_scale[:-1],
            central_diagonal_pixel_width)
        ]), 0, M - 1)

    return sparse.coo_matrix((values, (x_indices, y_indices)), shape=(M, N))    


class _BoxFactory(object):

    """
    For producing binning (box) matrices quickly on the fly.
    """

    def __init__(self, to_wavelengths, from_wavelengths, linear_tolerance=1e-3):

        self.to_wavelengths = to_wavelengths
        self.from_wavelengths = from_wavelengths
        self.N, self.M = (to_wavelengths.size, from_wavelengths.size)
        self._scale = self.M/np.ptp(self.from_wavelengths)
        if not linear_tolerance >= np.std(np.diff(from_wavelengths)):
            logger.warn("from wavelengths scale might be non-linear")


    @lru_cache(maxsize=LRU_SIZE, tol=6)
    def __call__(self, z=0, **kwargs):
        """
        Return a binning matrix for the given redshift based on the original
        wavelengths provided when the class was initiated.
        """
        """
        wavelengths = self.to_wavelengths * (1 + z)

        # Map which pixels go to which index, assuming that the from_wavelengths
        # scale is linear.
        pixel_edges = np.hstack([
            wavelengths[0] - (wavelengths[1] - wavelengths[0])/2.,
            wavelengths[:-1] + np.diff(wavelengths)/2.,
            wavelengths[-1] + (wavelengths[-1] - wavelengths[-2])/2.])

        fractional_indices_scale = (pixel_edges - self.from_wavelengths[0]) \
            * self._scale
        int_fractional_indices_scale = fractional_indices_scale.astype(int)

        upper_diagonals = 1 - fractional_indices_scale[:-1] % 1
        lower_diagonals = fractional_indices_scale[1:] % 1
        central_diagonal_pixel_width = np.diff(int_fractional_indices_scale) - 1

        normalised_scale = 1.0/(upper_diagonals + central_diagonal_pixel_width \
            + lower_diagonals)

        # If the to_wavelength is linear then the normalised scale per pixel
        # would be the same for every pixel. Let's not assume that..
        px_normalised_scale = np.repeat(normalised_scale,
            central_diagonal_pixel_width)

        values = np.hstack([
            upper_diagonals * normalised_scale,
            lower_diagonals * normalised_scale,
            np.ones(central_diagonal_pixel_width.sum()) * px_normalised_scale])

        _ = np.arange(self.N)
        y_indices = np.clip(
            np.hstack([_, _, np.repeat(_, central_diagonal_pixel_width)]),
                0, self.N - 1)
        x_indices = np.clip(np.hstack([
            int_fractional_indices_scale[:-1],
            int_fractional_indices_scale[1:],
            1 + np.repeat(int_fractional_indices_scale[:-1],
                central_diagonal_pixel_width)
            ]), 0, self.M - 1)

        return sparse.coo_matrix((values, (x_indices, y_indices)),
            shape=(self.M, self.N))
        """

        new_wavelengths = self.to_wavelengths * (1 + z)

        data = []
        old_px_indices = []
        new_px_indices = []
        for i, new_wl_i in enumerate(new_wavelengths):

            # These indices should span just over the new wavelength pixel.
            indices = np.unique(np.clip(
                self.from_wavelengths.searchsorted(new_wavelengths[i:i + 2], side="left")\
                    + [-1, +1], 0, self.from_wavelengths.size - 1))
            N = np.ptp(indices)

            if N == 0:
                # 'Fake' pixel.
                data.append(np.nan)
                new_px_indices.append(i)
                old_px_indices.extend(indices)
                continue

            # Sanity checks.
            assert (self.from_wavelengths[indices[0]] <= new_wl_i \
                or indices[0] == 0)
            assert (new_wl_i <= self.from_wavelengths[indices[1]] \
                or indices[1] == self.from_wavelengths.size - 1)

            fractions = np.ones(N)

            # Edges are handled as fractions between rebinned pixels.
            _ = np.clip(i + 1, 0, new_wavelengths.size - 1)
            lhs = self.from_wavelengths[indices[0]:indices[0] + 2]
            rhs = self.from_wavelengths[indices[-1] - 1:indices[-1] + 1]
            fractions[0]  = (lhs[1] - new_wavelengths[i])/np.ptp(lhs)
            fractions[-1] = (new_wavelengths[_] - rhs[0])/np.ptp(rhs)

            # Being binned to a single pixel. Prevent overflow from fringe cases.
            fractions = np.clip(fractions, 0, 1)
            fractions /= fractions.sum()

            data.extend(fractions) 
            new_px_indices.extend([i] * N) # Mark the new pixel indices affected.
            old_px_indices.extend(np.arange(*indices)) # And the old pixel indices.

        return sparse.csc_matrix((data, (old_px_indices, new_px_indices)),
            shape=(self.from_wavelengths.size, new_wavelengths.size))



class _BlurryBoxFactory(object):

    """
    For producing convolution (Blurring) and rebinning (Box) matrices quickly on
    the fly.
    """

    def __init__(self, to_wavelengths, from_wavelengths, from_resolution=None,
        threshold=5):

        self.to_wavelengths = to_wavelengths
        self.from_wavelengths = from_wavelengths

        self.from_resolution = from_resolution
        self._threshold_pixels = threshold/np.diff(self.from_wavelengths).max()

        self.N, self.M = (to_wavelengths.size, from_wavelengths.size)
        

    @lru_cache(maxsize=LRU_SIZE, tol=[0, 6])
    def __call__(self, resolution, z=0, **kwargs):
        """
        Return a binning matrix for the given resolution and optional redshift,
        based on the original wavelengths provided when the class was initiated.

        :param resolution:
            Spectral resolving power.

        :param z: [optional]
            The redshift.
        """

        # Calculate the redshifted wavelengths.
        wavelengths = self.to_wavelengths * (1 + z)

        # Calculate the width of the kernel at each point.
        fwhms = (wavelengths/resolution)**2
        if self.from_resolution:
            fwhms -= (wavelengths/self.from_resolution)**2

        # 2.355 ~= 2 * sqrt(2*log(2))
        sigmas = fwhms/2.3548200450309493
        N_kernel_pixels = int(np.ceil(sigmas[-1] * self._threshold_pixels))
        integer_offsets = self.from_wavelengths.searchsorted(wavelengths)

        # For +/- N_kernel_pixels at each point, calculate the kernel and retain
        # the indices.
        ios = integer_offsets[0]
        indices = np.tile(np.arange(ios - N_kernel_pixels,
            ios + N_kernel_pixels), self.N).reshape(self.N, 2 * N_kernel_pixels)
        indices[1:, :] += np.cumsum(np.diff(integer_offsets)).reshape(-1, 1)
        indices = np.clip(indices, 0, self.M - 1)

        x_indices = indices.flatten()
        y_indices = np.repeat(np.arange(self.N), 2 * N_kernel_pixels)

        # Calculate the kernel at each point.
        pdf = np.exp(
            -(self.from_wavelengths[indices].T - wavelengths)**2/(2*sigmas**2))
        pdf /= pdf.sum(axis=0)

        return sparse.coo_matrix((pdf.T.flatten(), (x_indices, y_indices)),
            shape=(self.M, self.N))

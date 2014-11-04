# coding: utf-8

""" Test specutils """

from __future__ import print_function

import os
import numpy as np

import unittest
import sick.specutils as specutils


class TestSpectrum1D(unittest.TestCase):

    def test_specutils_init_from_arrays(self):

        disp = np.arange(100)
        flux = np.random.uniform(size=100)
        variance = np.abs(np.random.uniform(size=100))

        spec = specutils.Spectrum1D(disp, flux, variance)
        return True

    def test_specutils_init_from_arrays_with_headers(self):

        disp = np.arange(100)
        flux = np.random.uniform(size=100)
        variance = np.abs(np.random.uniform(size=100))

        spec = specutils.Spectrum1D(disp, flux, variance, headers={"what": 3})
        return True

    def test_specutils_init_with_no_dispersion(self):
        self.assertRaises(ValueError, specutils.Spectrum1D, *[[], []])

    def test_specutils_init_with_disp_flux_mismatch(self):
        self.assertRaises(ValueError, specutils.Spectrum1D, *[[1], [2,3]])

    def test_specutils_copy(self):
        disp = np.arange(100)
        flux = np.random.uniform(size=100)
        variance = np.abs(np.random.uniform(size=100))

        spec = specutils.Spectrum1D(disp, flux, variance, headers={"what": 3})
        
        spec_copy = spec.copy()
        self.assertIsNone(np.testing.assert_allclose(spec.disp, spec_copy.disp))
        self.assertIsNone(np.testing.assert_allclose(spec.flux, spec_copy.flux))
        self.assertIsNone(np.testing.assert_allclose(spec.variance, spec_copy.variance))
        self.assertIsNone(np.testing.assert_allclose(spec.ivariance, spec_copy.ivariance))

        spec.disp[2] = 0
        spec.flux[2] = 0
        spec.variance[2] = 0
        spec.ivariance[2] = 0
        self.assertNotEqual(spec.disp[2], spec_copy.disp[2])
        self.assertNotEqual(spec.flux[2], spec_copy.flux[2])
        self.assertNotEqual(spec.variance[2], spec_copy.variance[2])
        self.assertNotEqual(spec.ivariance[2], spec_copy.ivariance[2])


    def test_specutils_save_load(self):

        disp = np.arange(100)
        flux = np.random.uniform(size=100)

        spec = specutils.Spectrum1D(disp, flux, headers={"this": "test"})
        spec.save("test.fits")
        spec.save("test.txt")

        spec_fits = specutils.Spectrum1D.load("test.fits")
        spec_txt = specutils.Spectrum1D.load("test.txt")

        self.assertIsNone(np.testing.assert_allclose(spec.disp, spec_fits.disp))
        self.assertIsNone(np.testing.assert_allclose(spec.flux, spec_fits.flux))
        self.assertIsNone(np.testing.assert_allclose(spec.variance, spec_fits.variance))
        self.assertEqual(0, len(set(map(str.lower, spec.headers.keys()))\
            .difference(map(str.lower, spec_fits.headers.keys()))))

        self.assertIsNone(np.testing.assert_allclose(spec.disp, spec_txt.disp))
        self.assertIsNone(np.testing.assert_allclose(spec.flux, spec_txt.flux))
        self.assertIsNone(np.testing.assert_allclose(spec.variance, spec_txt.variance))
        
        map(os.unlink, ["test.fits", "test.txt"])


    def runTest(self):
        pass
        

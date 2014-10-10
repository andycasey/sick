# coding: utf-8

""" Example inference using the sick command line interface """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@ast.cam.ac.uk>"

import os
import unittest
import urllib
import numpy as np
import subprocess
from glob import glob

import sick

np.random.seed(888)

# This world needs a little more truthiness.
truth = {
    "teff": 5454,
    "logg": 4.124,
    "feh": -0.514,
    "alpha": 0.02,
    "convolve.blue": 0.581,
    "z.blue": 13./299792458e-3,  # My lucky number.
    "f.blue": np.log(0.10), # ~10% underestimated
    "normalise.blue.c0": 1.63e-06,
    "normalise.blue.c1": -0.000788,
    "normalise.blue.c2": -0.000756,
}

TEST_DATA_URL = "http://astrowizici.st/test-inference-data.tar.gz"

class InferenceTest(unittest.TestCase):

    def setUp(self):
        """
        Download the model information and initialise it.
        """

        # Download the data that we need
        urllib.urlretrieve(TEST_DATA_URL, "test-inference-data.tar.gz")

        # Uncompress the data
        os.system("gunzip -f test-inference-data.tar.gz")
        os.system("tar -xzf test-inference-data.tar")

        # Initialise the model
        self.model = sick.models.Model("inference-model.yaml")

        # We create a faux-faux observation just so our faux observations get mapped
        # back onto the model.dispersion once they have been redshifted
        faux_obs = [sick.specutils.Spectrum1D(disp=self.model.dispersion[channel],
            flux=np.zeros(len(self.model.dispersion[channel]))) \
            for channel in self.model.channels]
        fluxes = self.model(observations=faux_obs, **truth)

        for i, (channel, flux) in enumerate(zip(self.model.channels, fluxes)):
            disp = self.model.dispersion[channel]

            N = len(disp)
            flux_err = 0.1 + 0.5 * np.random.randn(N)
            jitter_true = np.exp(truth["f.{0}".format(channel)])

            flux += np.abs(jitter_true*flux) * np.random.randn(N)
            flux += flux_err * np.random.randn(N)

            # Now let's throw away half of the data just for fun
            spectrum = sick.specutils.Spectrum1D(disp=disp[::2], flux=flux[::2],
                variance=flux_err[::2]**2)
            spectrum.save("sick-spectrum-{0}.fits".format(channel))
        return None


    def runTest(self):
        """
        Create a faux spectrum then infer the model parameters given the faux data.
        """

        executable = "sick solve inference-model.yaml".split()
        executable.extend(glob("sick-spectrum-*.fits"))

        output = subprocess.call(executable, env=os.environ.copy())
        assert output == 0
        return None


    def tearDown(self):
        """
        Remove the downloaded files, and remove the created figures.
        """

        # Remove the plots and things we produced
        filenames = glob("sick-spectrum-*")

        # Remove the model filenames
        filenames.extend(["inference-model.yaml", "inference-dispersion.memmap",
            "inference-flux.memmap", "inference-grid-points.pickle", 
            "test-inference-data.tar"])

        map(os.unlink, filenames)
        return None

if __name__ == "__main__":

    # Coveralls will run InferenceTest() properly, but sometimes the user might want to
    # run this themselves. If that's the case, we will not do the cleanup so that they
    # can look at the plots.
    dat_inference = InferenceTest()
    dat_inference.setUp()
    dat_inference.runTest()

    # If we are running this as main then clean up can be left as an exercise
    # for the reader
    #dat_inference.tearDown()
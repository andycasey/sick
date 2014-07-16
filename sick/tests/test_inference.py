# coding: utf-8

""" Example inference using sick """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@ast.cam.ac.uk>"

import os
import unittest
import urllib

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sick
import triangle

TEST_DATA_URL = "http://astrowizici.st/test-inference-data.tar.gz"

# This world needs a little more truthiness.
truth = {
    "teff": 5454,
    "logg": 4.124,
    "feh": -0.514,
    "alpha": 0.02,
    "convolve.blue": 0.581,
    "z.blue": +13.0/299792458e-3,
    "f.blue": np.log(0.10), # ~10% underestimated
    "normalise.blue.c0": 0.00123,
    "normalise.blue.c1": -0.5934,
    "normalise.blue.c2": -423.18,
}

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


    def runTest(self, acceptable_ci_multiple=None):
        """
        Create a faux spectrum then infer the model parameters given the faux data.
        """

        # Initialise the model
        model = sick.models.Model("inference-model.yaml")

        # We create a faux-faux observation just so our faux observations get mapped
        # back onto the model.dispersion once they have been redshifted
        faux_obs = [sick.specutils.Spectrum1D(disp=model.dispersion[channel],
            flux=np.zeros(len(model.dispersion[channel]))) \
            for channel in model.channels]
        fluxes = model(observations=faux_obs, **truth)

        observations = []
        for channel, flux in zip(model.channels, fluxes):
            disp = model.dispersion[channel]

            N = len(disp)
            flux_err = 0.1 + 0.5 * np.random.randn(N)
            jitter_true = np.exp(truth["f.{0}".format(channel)])

            flux += np.abs(jitter_true*flux) * np.random.randn(N)
            flux += flux_err * np.random.randn(N)

            # Now let's throw away half of the data just for fun
            spectrum = sick.specutils.Spectrum1D(disp=disp[::2], flux=flux[::2],
                variance=flux_err[::2]**2)
            observations.append(spectrum)

        # Plot the noisy spectrum
        fig, axes = plt.subplots(len(observations))
        if len(observations) == 1: axes = [axes]
        for ax, spectrum in zip(axes, observations):
            ax.plot(spectrum.disp, spectrum.flux, 'k')
            ax.set_ylabel("Flux, $F_\lambda$")
            ax.set_yticklabels([])
            ax.set_xlim(spectrum.disp[0], spectrum.disp[-1])
            ax.set_xlabel("Wavelength, $\lambda$ [$\AA$]")
        fig.savefig("spectrum.pdf")

        # Now let's solve for the model parameters
        posteriors, sampler, info = sick.solve(observations, model)

        # Plot the chains
        fig = sick.plot.chains(info["chain"], labels=sick.utils.latexify(model.parameters),
            truths=[truth[parameter] for parameter in model.parameters], burn_in=1000)
        fig.savefig("chains.pdf")

        # Make a corner plot with just the parameters of interest
        psi_len = len(model.grid_points.dtype.names)
        fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters))[:, :psi_len],
            labels=sick.utils.latexify(model.grid_points.dtype.names), quantiles=[.16, .50, .84], verbose=False,
            truths=[truth[parameter] for parameter in model.parameters[:psi_len]], extents=[0.95]*psi_len)
        fig.savefig("inference.pdf")

        # Make a corner plot with *all* of the model parameters
        fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters)),
            labels=sick.utils.latexify(model.parameters), quantiles=[.16, .50, .84], verbose=False,
            truths=[truth[parameter] for parameter in model.parameters])
        fig.savefig("inference-all.pdf")

        # Make a projection plot
        fig = sick.plot.projection(sampler, model, observations)
        fig.savefig("projection.pdf")

        # Assert that we have at least some solution
        if acceptable_ci_multiple is not None:
            for parameter in model.parameters:
                peak_posterior, pos_ci, neg_ci = posteriors[parameter]
                assert (peak_posterior + acceptable_ci_multiple * pos_ci >= truth[parameter] >= peak_posterior - acceptable_ci_multiple * neg_ci), (
                    "Inferences on the test case were not within {0}-'sigma' of the {1} truth values".format(acceptable_ci_multiple, parameter))
        

    def tearDown(self):
        """
        Remove the downloaded files, and remove the created figures.
        """

        # Remove the plots we produced
        filenames = ["chains.pdf", "spectrum.pdf", "inference.pdf", "inference-all.pdf", "projection.pdf"]

        # Remove the model filenames
        filenames.extend(["inference-model.yaml", "inference-dispersion.memmap", "inference-flux.memmap",
            "inference-grid-points.pickle", "test-inference-data.tar"])

        map(os.unlink, filenames)


if __name__ == "__main__":

    # Coveralls will run InferenceTest() properly, but sometimes the user might want to
    # run this themselves. If that's the case, we will not do the cleanup so that they
    # can look at the plots.
    dat_inference = InferenceTest()
    dat_inference.setUp()
    dat_inference.runTest()
    # Clean up can be left as an exercise for the reader
    #dat_inference.tearDown()
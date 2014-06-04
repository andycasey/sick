# coding: utf-8

""" Example inference using sick """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@ast.cam.ac.uk>"

import gzip
import os
import unittest
import urllib

import numpy as np
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
    "convolve.blue": 0.34,
    "z.blue": -56.12/299792458e-3,
    "jitter.blue": np.log(0.10), # ~10% underestimated
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
        os.system("gunzip test-inference-data.tar.gz")
        os.system("tar -xzf test-inference-data.tar")


    def runTest(self):
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
            jitter_true = np.exp(truth["jitter.{}".format(channel)])

            flux += np.abs(jitter_true*flux) * np.random.randn(N)
            flux += flux_err * np.random.randn(N)

            # Now let's throw away half of the data just for fun
            spectrum = sick.specutils.Spectrum1D(disp=disp[::2], flux=flux[::2],
                variance=flux_err[::2]**2)
            observations.append(spectrum)

        # Now let's solve for the model parameters
        posteriors, sampler, info = sick.solve(observations, model)

        # Plot the chains
        ndim = len(model.dimensions)
        chain_to_plot = sampler.chain.reshape(-1, ndim)
        chains_per_plot = len(model.grid_points.dtype.names)

        self.n, subplots = 1, 4
        steps = model.configuration["solver"]["burn"] + model.configuration["solver"]["sample"]
        for j, dimension in enumerate(model.dimensions):

            if j % subplots == 0:
                if j > 0:
                    [axes.xaxis.set_ticklabels([]) for axes in fig.axes[:-1]]
                    ax.set_xlabel("Iteration")
                    fig.savefig("chain-{0}.pdf".format(self.n))
                    self.n += 1
                fig = plt.figure()

            ax = fig.add_subplot(subplots, 1, (1 + j) % subplots)
            for k in range(model.configuration["solver"]["walkers"]):
                ax.plot(range(1, 1 + len(info["mean_acceptance_fractions"])),
                    info["chain"][k, :, j], c="k", alpha=0.5)
            ax.axvline(model.configuration["solver"]["burn"], ymin=0, ymax=1,
                linestyle=":", c="k")
            ax.set_ylabel(sick.utils.latexify([dimension])[0])
            ax.yaxis.set_major_locator(MaxNLocator(4))
    
            # Plot the truth
            ax.plot([0, steps], [truth[dimension], truth[dimension]], lw=2, c="#4682b4", zorder=10)
            ax.set_xlim(0, steps)

        [axes.xaxis.set_ticklabels([]) for axes in fig.axes[:-1]]
        ax.set_xlabel("Iteration")
        fig.savefig("chain-{0}.pdf".format(self.n))

        # Make a corner plot with just the parameters of interest
        psi_len = len(model.grid_points.dtype.names)
        fig = triangle.corner(sampler.chain.reshape(-1, len(model.dimensions))[:, :psi_len],
            labels=sick.utils.latexify(model.grid_points.dtype.names), quantiles=[.16, .50, .84], verbose=False,
            truths=[truth[dimension] for dimension in model.dimensions[:psi_len]], extents=[0.95]*psi_len)
        fig.savefig("inference.pdf")

        # Make a corner plot with *all* of the model parameters
        fig = triangle.corner(sampler.chain.reshape(-1, len(model.dimensions)),
            labels=sick.utils.latexify(model.dimensions), quantiles=[.16, .50, .84], verbose=False,
            truths=[truth[dimension] for dimension in model.dimensions])
        fig.savefig("inference-all.pdf")

        # Assert that we have found a solution at least within 2-sigma
        acceptable_ci_multiple = 2.
        for dimension in model.dimensions:
            peak_posterior, pos_ci, neg_ci = posteriors[dimension]
            assert (peak_posterior + acceptable_ci_multiple * pos_ci >= truth[dimension] >= peak_posterior - acceptable_ci_multiple * neg_ci), (
                "Inferences on the test case were not within {0}-sigma of the truth values".format(acceptable_ci_multiple))


    def tearDown(self):
        """
        Remove the downloaded files, and remove the created figures.
        """

        # Remove the plots we produced
        filenames = ["chain-{0}.pdf".format(n) for n in xrange(1, self.n+1)]
        filenames.extend(["inference.pdf", "inference-all.pdf"])

        # Remove the model filenames
        filenames.extend(["inference-model.yaml", "inference-dispersion.memmap", "inference-flux.memmap",
            "inference-grid-points.pickle", "test-inference-data.tar"])

        map(os.unlink, filenames)


if __name__ == "__main__":
    dat_inference = InferenceTest()
    dat_inference.setUp()
    dat_inference.runTest()
    dat_inference.tearDown()
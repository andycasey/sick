# coding: utf-8

""" Example inference using sick """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@ast.cam.ac.uk>"

import unittest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sick
import triangle

# This world needs a little more truthiness.
truth = {
    "teff": 3810,
    "logg": 0.403,
    "feh": -0.514,
    "alpha": 0.12,
    "convolve.blue": 1.734,
    "convolve.red": 0.671,
    "z.blue": 56.12/299792458e-3,
    "z.red": 64.34/299792458e-3,
    "jitter.blue": -2.321, # ~ 10% underestimated
    "jitter.red": -5.418,  # < 1% underestimated
    "normalise.blue.c0": 0.00123,
    "normalise.blue.c1": -0.5934,
    "normalise.blue.c2": -423.18,
    "normalise.red.c0": -0.00837,
    "normalise.red.c1": 49.741,
    "normalise.red.c2": -212763.37,
}

@unittest.skip("Model information not hosted online yet.")
class InferenceTest(unittest.TestCase):

    def setUp(self):
        """
        Download the model information and initialise it.
        """

        # TODO: Download the model information

        self.model = sick.models.Model("model.yaml")


    def runTest(self):
        """
        Create a faux spectrum then infer the model parameters given the faux data.
        """

        # We create a faux-faux observation just so our faux observations get mapped
        # back onto the model.dispersion once they have been redshifted
        faux_obs = [sick.specutils.Spectrum1D(disp=self.model.dispersion[channel],
            flux=np.zeros(len(self.model.dispersion[channel]))) \
            for channel in self.model.channels]
        fluxes = self.model(observations=faux_obs, **truth)

        observations = []
        for channel, flux in zip(self.model.channels, fluxes):
            disp = self.model.dispersion[channel]

            N = len(disp)
            flux_err = 0.1 + 0.5 * np.random.randn(N)
            jitter_true = np.exp(truth["jitter.{}".format(channel)])

            flux += np.abs(jitter_true*flux) * np.random.randn(N)
            flux += flux_err * np.random.randn(N)

            # Apply the Pb, Vb

            # Throw away half of the data. Just for fun.
            spectrum = sick.specutils.Spectrum1D(disp=disp[::2], flux=flux[::2],
                variance=flux_err[::2]**2)
            observations.append(spectrum)

        # Now let's solve for the self.model parameters
        posteriors, sampler, info = sick.solve(observations, self.model)

        # Plot the chains
        ndim = len(self.model.dimensions)
        chain_to_plot = sampler.chain.reshape(-1, ndim)
        chains_per_plot = len(self.model.grid_points.dtype.names)

        self.n, subplots = 1, 4
        for j, dimension in enumerate(self.model.dimensions):

            if j % subplots == 0:
                if j > 0:
                    [axes.xaxis.set_ticklabels([]) for axes in fig.axes[:-1]]
                    ax.set_xlabel("Iteration")
                    fig.savefig("chain-{0}.pdf".format(self.n))
                    self.n += 1
                fig = plt.figure()

            ax = fig.add_subplot(subplots, 1, (1 + j) % subplots)
            for k in range(self.model.configuration["solver"]["walkers"]):
                ax.plot(range(1, 1 + len(info["mean_acceptance_fractions"])),
                    info["chain"][k, :, j], c="k", alpha=0.5)
            ax.axvline(self.model.configuration["solver"]["burn"], ymin=0, ymax=1,
                linestyle=":", c="k")
            ax.set_ylabel(sick.utils.latexify([dimension])[0])
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlim(0, ax.get_xlim()[1])

        [axes.xaxis.set_ticklabels([]) for axes in fig.axes[:-1]]
        ax.set_xlabel("Iteration")
        fig.savefig("chain-{0}.pdf".format(self.n))

        # Make a corner plot with just the parameters of interest
        psi_len = len(self.model.grid_points.dtype.names)
        fig = triangle.corner(sampler.chain.reshape(-1, len(self.model.dimensions))[:, :psi_len],
            labels=sick.utils.latexify(self.model.grid_points.dtype.names), quantiles=[.16, .50, .84], verbose=False,
            truths=[truth[dimension] for dimension in self.model.dimensions[:psi_len]], extents=[0.95]*psi_len)
        fig.savefig("inference.pdf")

        # Make a corner plot with *all* of the self.model parameters
        fig = triangle.corner(sampler.chain.reshape(-1, len(self.model.dimensions)),
            labels=sick.utils.latexify(self.model.dimensions), quantiles=[.16, .50, .84], verbose=False,
            truths=[truth[dimension] for dimension in self.model.dimensions])
        fig.savefig("inference-all.pdf")

        # Assert that we have found a 'pretty good' solution.
        acceptable_ci_multiple = 1.5
        for dimension in self.model.dimensions:
            peak_posterior, pos_ci, neg_ci = posteriors[dimension]
            assert (peak_posterior + acceptable_ci_multiple * pos_ci >= truth[dimension] >= peak_posterior + acceptable_ci_multiple * neg_ci)


    def tearDown(self):
        """
        Remove the downloaded files, and remove the created figures.
        """

        filenames = ["chain-{0}.pdf".format(n) for n in xrange(1, self.n+1)]
        filenames.extend(["inference.pdf", "inference-all.pdf"])

        filenames.extend(["model.yaml", "grid-points.pickle", "dispersion.memmap", "fluxes.memmap"])

        map(os.unlink, filenames)

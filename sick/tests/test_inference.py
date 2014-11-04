# coding: utf-8

""" Example inference using sick """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@ast.cam.ac.uk>"

import os
import unittest
import urllib
import numpy as np
from glob import glob

import sick
import sick.cli

np.random.seed(888)

# This world needs a little more truthiness.
truth = {
    "teff": 5454,
    "logg": 4.124,
    "feh": -0.514,
    "alpha": 0.02,
    "convolve.blue": 0.581,
    "z.blue": np.random.normal(0, 100)/299792.458,
    "normalise.blue.c0": 1.63e-06,
    "normalise.blue.c1": -0.000788,
    "normalise.blue.c2": -0.000756,
}

TEST_DATA_URL = "http://astrowizici.st/test-inference-data.tar.gz"

class InferenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Download the model information and initialise it.
        """

        # Download the data that we need
        if not os.path.exists("inference-model.yaml"):
            urllib.urlretrieve(TEST_DATA_URL, "test-inference-data.tar.gz")

            # Uncompress the data
            os.system("gunzip -f test-inference-data.tar.gz")
            os.system("tar -xzf test-inference-data.tar")

        else:
            print("DATA FOUND ALREADY.")

        cls.model = sick.models.Model("inference-model.yaml")

        # We create a faux-faux observation just so our faux observations get 
        # mapped back onto the model.dispersion once they have been redshifted
        faux_obs = [sick.specutils.Spectrum1D(disp=cls.model.dispersion[c],
            flux=np.zeros(len(cls.model.dispersion[c]))) \
                for c in cls.model.channels]
        fluxes = cls.model(data=faux_obs, **truth)

        for i, (channel, flux) in enumerate(zip(cls.model.channels, fluxes)):
            
            disp = cls.model.dispersion[channel]
            flux = flux.copy()

            N = len(disp)
            flux_err = np.random.poisson(flux, size=flux.size)**0.5
            flux += flux_err * np.random.randn(N)
            
            spectrum = sick.specutils.Spectrum1D(disp=disp, flux=flux,
                variance=flux_err**2)
            spectrum.save("sick-spectrum-{0}.fits".format(channel))


    def test_api(self):
        """
        Create a faux spectrum then infer the model parameters given the data.
        """

        # Initialise the model
        model = sick.models.Model("inference-model.yaml")
        data = map(sick.specutils.Spectrum1D.load, 
            ["sick-spectrum-{0}.fits".format(c) for c in self.model.channels])

        # Now let's solve for the model parameters
        optimised_theta, optimised_r_chi_sq, optimised_info = model.optimise(
            data, fixed=["z.{}".format(c) for c in model.channels])

        posteriors, sampler, info = model.infer(data, theta=optimised_theta)

        # Plot the chains
        fig = sick.plot.chains(info["chain"],
            labels=sick.utils.latexify(model.parameters),
            burn_in=model.configuration["settings"]["burn"],
            truths=[truth[p] for p in model.parameters])
        fig.savefig("chains-api.pdf")

        # Make a corner plot with just the parameters of interest
        psi_len = len(model.grid_points.dtype.names)
        fig = sick.plot.corner(
            sampler.chain.reshape(-1, len(model.parameters))[:, :psi_len],
            labels=sick.utils.latexify(model.grid_points.dtype.names), 
            truths=[truth[p] for p in model.parameters[:psi_len]],
            quantiles=[.16, .50, .84], verbose=False)
        fig.savefig("inference-api.pdf")

        # Make a corner plot with *all* of the model parameters
        fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters)),
            labels=sick.utils.latexify(model.parameters), 
            truths=[truth[p] for p in model.parameters],
            quantiles=[.16, .50, .84], verbose=False)
        fig.savefig("inference-all-api.pdf")

        # Make a projection plot
        fig = sick.plot.projection(model, data, chain=sampler.chain)
        fig.savefig("projection-api.pdf")

        # Make an auto-correlation plot
        fig = sick.plot.autocorrelation(sampler.chain)
        fig.savefig("autocorrelation-api.pdf")

        # Make a mean acceptance fraction plot
        fig = sick.plot.acceptance_fractions(info["mean_acceptance_fractions"],
            burn_in=model.configuration["settings"]["burn"])
        fig.savefig("acceptance-api.pdf")


    def test_cli(self):
        executable = "solve inference-model.yaml".split()
        executable.extend(["sick-spectrum-{}.fits".format(c) \
            for c in self.model.channels])
        print("Executing command: {}".format(executable))
        args = sick.cli.parser(executable)
        assert args.func(args)


    def runTest(self):
        pass


    @classmethod
    def tearDownClass(cls):
        """
        Remove the downloaded files, and remove the created figures.
        """

        # Remove the plots we produced
        filenames = ["chains.pdf", "inference.pdf", "acceptance.pdf",
            "inference-all.pdf", "projection.pdf", "autocorrelation.pdf",
            "chains-api.pdf", "inference-api.pdf", "acceptance-api.pdf",
            "inference-all-api.pdf", "projection-api.pdf", "autocorrelation-api.pdf"]
        filenames.extend(glob("sick-spectrum-blue*"))

        # Remove the model filenames
        filenames.extend(["inference-model.yaml", "inference-dispersion.memmap",
            "inference-flux.memmap", "inference-grid-points.pickle", 
            "test-inference-data.tar"])

        for filename in filenames:
            print("Removing filename {}".format(filename))
            if os.path.exists(filename):
                os.unlink(filename)
            else:
                print("Expected file {0} does not exist!".format(filename))


if __name__ == "__main__":

    # Coveralls will run InferenceTest() properly, but sometimes the user might 
    # want to run this themselves. If that's the case, we will not do the 
    # cleanup so that they can look at the plots.
    dat_inference = InferenceTest()
    dat_inference.setUpClass()
    dat_inference.test_cli()
    dat_inference.test_api()

    # So if we are running this as main then clean up can be left as an exercise
    # for the reader
    #dat_inference.tearDownClass()
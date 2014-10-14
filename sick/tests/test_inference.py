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
    "z.blue": 13./299792458e-3,  # My lucky number.
    "f.blue": np.log(0.10), # ~10% underestimated
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
        urllib.urlretrieve(TEST_DATA_URL, "test-inference-data.tar.gz")

        # Uncompress the data
        os.system("gunzip -f test-inference-data.tar.gz")
        os.system("tar -xzf test-inference-data.tar")

        model = sick.models.Model("inference-model.yaml")

        # We create a faux-faux observation just so our faux observations get mapped
        # back onto the model.dispersion once they have been redshifted
        faux_obs = [sick.specutils.Spectrum1D(disp=model.dispersion[channel],
            flux=np.zeros(len(model.dispersion[channel]))) \
            for channel in model.channels]
        fluxes = model(observations=faux_obs, **truth)

        for i, (channel, flux) in enumerate(zip(model.channels, fluxes)):
            disp = model.dispersion[channel]

            N = len(disp)
            flux_err = 0.1 + 0.5 * np.random.randn(N)
            jitter_true = np.exp(truth["f.{0}".format(channel)])

            flux += np.abs(jitter_true*flux) * np.random.randn(N)
            flux += flux_err * np.random.randn(N)

            # Now let's throw away half of the data just for fun
            spectrum = sick.specutils.Spectrum1D(disp=disp[::2], flux=flux[::2],
                variance=flux_err[::2]**2)
            spectrum.save("sick-spectrum-{0}.fits".format(channel))


    def runTest(self):
        self.run_api()
        self.run_cli()


    def run_api(self):
        """
        Create a faux spectrum then infer the model parameters given the faux data.
        """

        # Initialise the model
        model = sick.models.Model("inference-model.yaml")
        observations = map(sick.specutils.Spectrum.load, glob("sick-spectrum-*.fits"))

        # Now let's solve for the model parameters
        posteriors, sampler, info = sick.solve(observations, model)
        print(posteriors)

        # Plot the chains
        fig = sick.plot.chains(info["chain"],
            labels=sick.utils.latexify(model.parameters), burn_in=1000,
            truths=[truth[parameter] for parameter in model.parameters])
        fig.savefig("chains.pdf")

        # Make a corner plot with just the parameters of interest
        psi_len = len(model.grid_points.dtype.names)
        fig = sick.plot.corner(
            sampler.chain.reshape(-1, len(model.parameters))[:, :psi_len],
            labels=sick.utils.latexify(model.grid_points.dtype.names), 
            truths=[truth[parameter] for parameter in model.parameters[:psi_len]],
            quantiles=[.16, .50, .84], verbose=False)
        fig.savefig("inference.pdf")

        # Make a corner plot with *all* of the model parameters
        fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters)),
            labels=sick.utils.latexify(model.parameters), 
            truths=[truth[parameter] for parameter in model.parameters],
            quantiles=[.16, .50, .84], verbose=False)
        fig.savefig("inference-all.pdf")

        # Make a projection plot
        fig = sick.plot.projection(model, observations, sampler=sampler)
        fig.savefig("projection.pdf")

        # Make an auto-correlation plot
        fig = sick.plot.autocorrelation(sampler.chain)
        fig.savefig("autocorrelation.pdf")

        # Make a mean acceptance fraction plot
        fig = sick.plot.acceptance_fractions(info["mean_acceptance_fractions"],
            burn_in=model.configuration["settings"]["burn"])
        fig.savefig("acceptance.pdf")


    def run_cli(self):
        executable = "solve inference-model.yaml".split()
        executable.extend(glob("sick-spectrum-*.fits"))
        args = sick.cli.parser(executable)
        output = args.func(args)
        assert output == True
        
        return None
        

    @classmethod
    def tearDownClass(cls):
        """
        Remove the downloaded files, and remove the created figures.
        """

        # Remove the plots we produced
        filenames = ["chains.pdf", "inference.pdf", "acceptance.pdf",
            "inference-all.pdf", "projection.pdf", "autocorrelation.pdf"]

        filenames.extend(glob("sick-spectrum-blue*"))

        # Remove the model filenames
        filenames.extend(["inference-model.yaml", "inference-dispersion.memmap",
            "inference-flux.memmap", "inference-grid-points.pickle", 
            "test-inference-data.tar"])

        map(os.unlink, filenames)


if __name__ == "__main__":

    # Coveralls will run InferenceTest() properly, but sometimes the user might want to
    # run this themselves. If that's the case, we will not do the cleanup so that they
    # can look at the plots.
    dat_inference = InferenceTest()
    dat_inference.setUpClass()
    dat_inference.run_cli()
    dat_inference.run_api()

    # If we are running this as main then clean up can be left as an exercise
    # for the reader
    #dat_inference.tearDown()
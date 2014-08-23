.. Getting started guide, which shows how to do the self-consistent inference test. 

===============
Getting Started
===============

This page will guide you through the example presented in Section 2.4 of the *sick* paper:

1. Downloading a cached model.
2. Generating a fake, noisy spectrum from a model grid.
3. Inferring the properties of the fake spectrum using *sick*.

If you simply want to run this test and skip the explanatory text below, you can do so with::

    cd /wherever/sick/is/kept
    nosetests -v

or from within Python::

    from sick.tests.test_inference import InferenceTest

    dat_inference = InferenceTest()
    dat_inference.setUp()
    dat_inference.runTest()
    
    # Now you will find a number of output plots in the current working directory
    # To remove these, use:
    dat_inference.tearDown()


Step 1: Downloading the cached model
------------------------------------

*sick* caches model spectra so that they can be accessed and interpolated quickly at run-time. Usually we would have to `cache the model
ourselves <caching_a_model>`_, but for this test we will just download a pre-cached subset of 
the `AMBRE spectral library <http://adsabs.harvard.edu/abs/2012A%26A...544A.126D>`_ with the following commands::

    wget http://astrowizici.st/test-inference-data.tar.gz 
    gunzip test-inference-data.tar.gz
    tar -xzf test-inference-data.tar

Now you should have the following files::

    inference-mdoel.yaml
    inference-flux.memmap
    inference-dispersion.memmap
    inference-grid-points.pickle

The model filename, ``inference-model.yaml``, contains the following information::

    cached_channels:
      points_filename: inference-grid-points.pickle
      flux_filename: inference-flux.memmap
      blue:
        dispersion_filename: inference-dispersion.memmap

    convolve:
      blue: yes

    redshift:
      blue: yes

    normalise:
      blue:
        method: polynomial
        order: 2

    settings:
      initial_samples: 1000
      burn: 1000
      sample: 500
      walkers: 200
      threads: 24

You can see that our model filename ``inference-model.yaml`` references the grid points in ``inference-grid-points.pickle``, the fluxes in ``inference-flux.memmap``, and the dispersion points in ``inference-dispersion.memmap``.

Step 2: Generate a fake, noisy spectrum
---------------------------------------

Now we'll create a fake spectrum from our model grid, add a bunch of noise, then try and
recover the stellar parameters. Here's the Python code::

    truth = {
        "teff": 5454,
        "logg": 4.124,
        "feh": -0.514,
        "alpha": 0.02,
    }



.. Getting started guide, which shows how to do the self-consistent inference test. 

=======================================
Example: Self-consistent inference test
=======================================

(This page will guide you through the example presented in Section 2.4 of the *sick* paper)

In this example we will download a cached subset of the `AMBRE stellar spectral library <http://adsabs.harvard.edu/abs/2012A%26A...544A.126D>`_,
interpolate a spectrum and turn it into a fake observation by introducing a continuum shape, smoothing it, redshifting it, and adding noise. 
Then we'll infer the model parameters for the fake spectrum and examine how well our inferences are. 

.. note::
   This example is part of the automatic testing suite in ``sick``, and is executed every time a change is commited to the `GitHub repository <https://github.com/andycasey/sick>`_. If you want to run this test and skip the explanatory text below, you can do so with::

       cd /wherever/sick/is/kept
       nosetests -v

   or from within Python::

       from sick.tests.test_inference import InferenceTest

       inference = InferenceTest()
       inference.setUp()
       inference.runTest()
    
       # Now you will find a number of output plots in the current working directory
       # To remove these, use:
       inference.tearDown()


Downloading the cached model
----------------------------

You can download a cached subset of the model grid by using the following commands:: 

    wget http://astrowizici.st/test-inference-data.tar.gz 
    gunzip test-inference-data.tar.gz
    tar -xzf test-inference-data.tar

Now you should have the following files in your current working directory::

    inference-model.yaml
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

You can see that our model filename ``inference-model.yaml`` references the grid points in ``inference-grid-points.pickle``, the fluxes in ``inference-flux.memmap``, and the dispersion points in ``inference-dispersion.memmap``. Now we're ready to create a fake observation.

Generate a fake spectrum
------------------------

Now we'll create a fake spectrum from our model grid, add a bunch of noise, then try and
recover the stellar parameters. Here's the Python code::

    import numpy as np
    import matplotlib.pyplot as plt

    import sick

    truth = {
        # Chose some stellar parameters
        "teff": 5454,
        "logg": 4.124,
        "feh": -0.514,
        "alpha": 0.02,

        # Blur the spectrum
        "convolve.blue": 0.581,

        # Redshift the spectrum
        "z.blue": +13.0/299792458e-3,

        # Underestimate the variance
        "f.blue": np.log(0.10), # ~10% underestimated

        # Create a third order polynomial continuum shape 
        "normalise.blue.c0": 0.00123,
        "normalise.blue.c1": -0.5934,
        "normalise.blue.c2": -423.18,
    }

    # Initialise the model
    model = sick.Model("inference-model.yaml")

    # This part is a little bit meta:
    # We create a faux-faux observation just so our faux observations get mapped
    # back onto the model.dispersion once they have been redshifted
    N = len(model.dispersion["blue"])
    faux_obs = [sick.specutils.Spectrum1D(disp=model.dispersion["blue"],
        flux=np.zeros(N))]

    # Create our fake looking, but still noise-free, spectrum.
    faux_flux = model(observations=faux_obs, **truth)[0]

    # OK now let's add noise 
    flux_err = 0.1 + 0.5 * np.random.randn(N)
    jitter_true = np.exp(truth["f.blue"])
    faux_flux += np.abs(jitter_true * faux_flux) * np.random.randn(N)
    faux_flux += flux_err * np.random.randn(N)

    # Combine our dispersion and flux into a 1D spectrum, and we'll throw away
    # half of the data (just for fun!)
    observed_data = sick.specutils.Spectrum1D(disp=model.dispersion["blue"][::2],
        flux=faux_flux[::2], variance=flux_err[::2]**2)

    # Let's save the fake spectrum 
    observed_data.save("faux_spectrum.fits")


Let's see what our fake spectrum looks like::

    fig, axes = plt.subplots()
    ax = axes[0]
    ax.plot(observed_data.disp, observed_data.flux, 'k')
    ax.set_xlabel("Wavelength, $\lambda$ [$\AA$]")
    ax.set_ylabel("Flux, $F_\lambda$")
    fig.savefig("spectrum.pdf")
   
     
% Figure 



Inference
---------

Now that we have our model and a (fake) spectrum we can do some inference! To continue doing this in Python::

    # Let's get cranking
    posteriors, sampler, info = sick.solve([observed_data], model)

Or we could use the command line tool directly from the terminal::

    sick solve inference-model.yaml faux_spectrum.fits

If you use the ``sick solve`` command line function then this will (by default) generate some post-processing 
plots for you, so you won't need to execute the code in the following section.


Post-Processing
---------------

Once the analysis is complete you will surely want to look at some plots to ensure everything has run smoothly.
Continuing in Python::

    # Plot the values of all the chains
    fig = sick.plot.chains(info["chain"], labels=sick.utils.latexify(model.parameters),
        truths=[truth[parameter] for parameter in model.parameters], burn_in=1000)
    fig.savefig("chains.pdf")

    # Make a corner plot with just the astrophysical parameters
    psi_len = len(model.grid_points.dtype.names)
    fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters))[:, :psi_len],
        labels=sick.utils.latexify(model.grid_points.dtype.names),
        truths=[truth[parameter] for parameter in model.parameters[:psi_len]],
        quantiles=[.16, .50, .84], verbose=False)
    fig.savefig("inference-psi.pdf")

    # Make a corner plot with *all* of the model parameters
    fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters)),
        labels=sick.utils.latexify(model.parameters), 
        truths=[truth[parameter] for parameter in model.parameters],
        quantiles=[.16, .50, .84], verbose=False)
    fig.savefig("inference-all.pdf")

    # Make a projection plot
    fig = sick.plot.projection(sampler, model, [observed_data])
    fig.savefig("projection.pdf")

% Figures

 

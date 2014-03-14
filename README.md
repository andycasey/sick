
SPECTROSCOPE
------------

[![Build Status](https://travis-ci.org/andycasey/spectroscope.png?branch=master)](https://travis-ci.org/andycasey/spectroscope) [![PyPi download count image](https://pypip.in/d/scope/badge.png)](https://pypi.python.org/pypi/spectroscope/)

**Author:** Andy Casey ([arc@ast.cam.ac.uk](mailto:arc@ast.cam.ac.uk))

**License:** Modified [Academic License](http://github.com/dfm/license): This project includes academic-research code and documents under development. You would be a fool to run any of the code. Please contact the author before using this code. 

Principle
---------
All other attempts to determine stellar parameters from a grid of
pre-computed spectra separate out the normalisation, radial velocity, and synthetic
smoothing components. In reality these phenomena are all linked, and the reliability
of stellar parameter inference will be affected by any uncertainty in these components.
You should have a single mathematical model that can incorporate all of these convolutions. 
That's what this code does. It's flexible enough for use on any type of spectra.

Installation
------------

``pip install spectroscope`` (or [if you must](https://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install), use ``easy_install spectroscope``)

Usage
-----
Running SCOPE should be as easy as:

``scope model.yml blue_spectrum.fits red_spectrum.fits``

If ``blue_spectrum.fits`` and ``red_spectrum.fits`` have multiple apertures (e.g., multiplexing) then all spectra will be analysed.

Model Example
-------------
In the usage example above, the ``model.yml`` file contains all the model information required. This file can be a YAML or JSON-style format. Below is an example of what ``model.yml`` might look like, with comments:

````
solver:
  method: emcee
  nwalkers: 200
  burn: 300
  sample: 200
  threads: 8

models:
  dispersion_filenames:
    # Here we are going to decide on the names of each 'aperture', or each portion of spectrum.
    # If you only have one complete spectrum, just call it 'optical' or something similar.
    blue: /media/wd/ges/synthetic-spectra/GES_UVESRed580/GES_UVESRed580_Lambda.fits
    red:  /media/wd/ges/synthetic-spectra/GES_HR21/GES_HR21_Lambda.fits

  flux_filenames:
    # Here all of our flux files are stored in a folder for each aperture we just named.
    # The parameters we want to infer are in the flux filenames. We can call these parameters
    # whatever we want, and we use a regular expression to find all the relevant files.
    blue:
      folder: "/media/wd/ges/synthetic-spectra/GES_UVESRed580/GES_UVESRed580_deltaAlphaFe+0.0.fits/"
      re_match: '[s|p](?P<teff>[0-9]+):g\+(?P<logg>[0-9.]+).+z(?P<feh>[0-9.+-]+).+fits'
    red: 
      folder: "/media/wd/ges/synthetic-spectra/GES_HR21/GES_HR21_deltaAlphaFe+0.0.fits/"
      re_match: '[s|p](?P<teff>[0-9]+):g\+(?P<logg>[0-9.]+).+z(?P<feh>[0-9.+-]+).+fits'

# Allow for normalisation in the observed spectra. This will add (order + 1) free parameters
# to our model for each aperture
normalise_observed:
  blue:
    perform: true
    order: 2
  red:
    perform: true
    order: 3

# Allow for individual doppler shifts in each aperture
doppler_shift:
  blue:
    perform: true
  red:
    perform: true

# Allow for each aperture to have different resolutions
smooth_model_flux:
  blue:
    perform: true
    kernel: free
  red:
    perform: true 
    kernel: free

# We will set explicit priors for our model dimensions
priors:
  # We are going to assume we know nothing about these stars a priori.
  # That means the initial guesses will be uniformly distributed random
  # guesses across the full extent of the parameter space.
  teff: uniform 
  logg: uniform 
  feh: uniform

  # For any star, the a priori heliocentric velocity probability is a
  # Gaussian centered on 0 km/s with a dispersion around 100 km/s
  doppler_shift.blue: normal(0, 100)
  doppler_shift.red: normal(0, 100)

  # The synthetic spectra have been pre-convolved to a lower resolution,
  # but not the exact resolution of our data. This is because the exact
  # amount of smoothing can vary slightly depending on the star.
  smooth_model_flux.blue.kernel: normal(3, 0.1)  # For original data
  smooth_model_flux.red.kernel: normal(0.4, 0.1) # For original data

  # Note that we have not specified any explicit priors for our
  # normalisation coefficients (normalise_observed.blue.a0, etc).
  # If not specified, the mu and sigma for a reasonable normally-distributed
  # prior will be determined automatically.

# Masks are optional. When they are present, these regions specify the
# regions to use for comparison. If no masks are specified, the entire
# spectral region overlapping the model and observed spectra is used.
masks:
  blue:
    - [4500, 5300]
  red:
    - [8450, 8750]
````


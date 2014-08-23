.. Getting started guide, which shows how to do the self-consistent inference test. 

*********************
Getting Started Guide
*********************

This guide will explain the practical concept of models in ``sick``, show you how to set up a model, guide you through how to cache the model for fast run-time access, demonstrate how to fit a model to some data, and perform post-processing of the results.


The Generative Model
--------------------

A generative model is a mathematical model with a number of free parameters that for some values of those parameters you can generate what your data would look like. For most processes in astrophysics where we are dealing with spectra, there are so many free parameters that this becomes a very expensive endeavour to do at run-time. Because of the expense, we usually pre-compute large grids of spectra for combinations of different model parameters. This is where ``sick`` comes in.

``sick`` allows you to use a pre-computed grid of spectra and efficiently interpolate a spectrum at any point within the grid. Additional phenomena that can effect the observed spectrum (e.g., redshift, continuum, outlier pixels) can be modelled simultaneously, allowing you to precisely infer all of the model parameters at once. We do this by specifying a **model configuration file**, where you can enable or disable any combination of these effects. The same model file can be used for thousands of spectra, allowing you to trivially fit a model to large spectral data sets. 

Creating the model configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step of this guide will be creating the model configuration file. For the purpose of this guide, I will assume that you have some grid of model spectra that is all sampled at the same dispersion (e.g., wavelength or frequency) points. I'll also assume that each computed spectrum is in a different file, and that the filename includes the model parameters that were used to calculate that spectrum. For example, you might have a file structure like this::

   $ pwd
   /synthetic-spectra/my-blue-spectral-grid/

   $ ls -l *.txt | tail -n 5
   spectrum_4000_0.5_-0.3_+0.4.txt
   spectrum_4000_0.5_-0.3_+0.4.txt
   spectrum_5000_0.5_-0.5_+0.4.txt
   spectrum_5000_0.5_-0.5_+0.4.txt
   wavelengths.txt

.. note::
   
   Don't worry if this file structure does not exactly match the format for your synthetic spectra. ``sick`` is reasonably flexible about how to create a model, and it's reasonably trivial to export data into a format that is acceptable to ``sick``.


Because all of the spectra are sampled at the same dispersion points, it makes sense to have those dispersion points saved in one file (``wavelengths.txt``) and the fluxes (or intensities) in different files. Each of the ``spectrum_*.txt`` files has numbers in it that represent values for some model parameter. For the moment we will call these four parameters :math:`\Theta = (A, B, C, D)`. You can name these parameters whatever you like; ``sick`` is ambivalent to *what* the parameters actually describe, but we do have to name them *something*.

The way we tell ``sick`` the names of the model parameters and where to find the files is by using `regular expressions <https://en.wikipedia.org/wiki/Regular_expression>`_. ``sick`` uses `named groups <https://docs.python.org/2/library/re.html>`_ to identify the model parameters and their values for each spectrum. If you haven't used regular expressions before, `RegExr <http://www.regexr.com/>`_ serves as a useful guide. However, Python group-matching in regular expressions is slightly different, so you might want to use `this website <http://www.pythonregex.com/>`_ to test your expressions. 

The model configuration file can be formatted in `YAML <http://www.yaml.org>`_ or `JSON <http://www.json.org>`_. Here's what a YAML-formatted configuration file might look like for the example above::

    # Define the spectral channel
    channels:
      blue:
        dispersion_filename: /synthetic-spectra/my-blue-spectral-grid/wavelengths.txt
        flux_folder: /synthetic-spectra/my-blue-spectral-grid/
        flux_filename_match: 'spectrum_(?P<alpha>[0-9]{4})_(?P<beta>[0-9.+-]+)_(?P<charlie>[0-9.+-]+)_(?P<delta>[0-9.+-]+)\.txt'

    # Specify some solver settings
    solver:
      optimise: yes
      walkers: 200
      initial_samples: 1000
      burn: 400
      sample: 100

    # Now specify some model settings
    normalise:
      blue:
        method: polynomial
        order: 3

    outliers: yes
    redshift: yes
    convolve: no


In addition to specifying some model and solver settings, I have defined a single channel named ``blue`` (you can name it anything you want) and you can see that I have defined the model parameters (in ``flux_filename_match``) as ``alpha``, ``beta``, ``charlie``, and ``delta``. You can also see that regular expression will only find ``alpha`` values between ``0000-9999``. If you get stuck with regular expressions, `drop me a line <mailto:arc@ast.cam.ac.uk>`_.


Caching the Model
-----------------

Once you have a configuraton file you can cache the model so that it's faster to read and interpolate spectra at run-time. The caching process loads all of the synthetic spectra and saves `C-contiguous <http://docs.scipy.org/doc/numpy/reference/internals.code-explanations.html>`_ `memory-mapped arrays <https://docs.python.org/2/library/mmap.html>`_ of the dispersion points, the model parameters for each spectrum, and the fluxes for each spectrum. Caching models is **strongly recommended**. 

In addition to being computationally efficient, caching a model allows you to have efficient synthetic grids that are far larger than the available random access memory. During the caching process you can also pre-convolve the spectra, re-sample it, or splice certain wavelength ranges. 

Using the command line tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``sick`` command line tool can be used to cache simple models where no pre-convolution, resampling, or wavelength splicing is required. If your model configuration filename is ``model.yaml`` then you can cache it using the command::

    sick cache model.yaml grid_points.pickle dispersion.memmap fluxes.memmap

This will update the ``model.yaml`` file with the cached model information and create the files ``grid_points.pickle``, ``dispersion.memmap``, and ``fluxes.memmap``. The original model information in ``model.yaml`` will be commented out, so the updated file will look something like::

    # Define the spectral channel
    #channels:
    #  blue:
    #    dispersion_filename: /synthetic-spectra/my-blue-spectral-grid/wavelengths.txt
    #    flux_folder: /synthetic-spectra/my-blue-spectral-grid/
    #    flux_filename_match: 'spectrum_(?P<alpha>[0-9]{4})_(?P<beta>[0-9.+-]+)_(?P<charlie>[0-9.+-]+)_(?P<delta>[0-9.+-]+)\.txt'

    channels:
      points_filename: grid_points.pickle
      flux_filename: fluxes.memmap

      blue:
         dispersion_filename: dispersion.memmap

    # Specify some solver settings
    solver:
      optimise: yes
      walkers: 200
      initial_samples: 1000
      burn: 400
      sample: 100

    # Now specify some model settings
    normalise:
      blue:
        method: polynomial
        order: 3

    outliers: yes
    redshift: yes
    convolve: no


.. note::

   The different structure in ``channels`` for the cached model is because when there are multiple channels, the fluxes for **all** channels are stored in ``flux_filename``. This allows for models with multiple channels where the total grid size exceeds the available random access memory.


Using the API
^^^^^^^^^^^^^

If you wish to pre-convolve, resample, or splice any of the model spectra while caching, then you will need to use the :py:func:`sick.models.Model.cache` function. 

.. code-block:: python

   import yaml # or json, in which case use json.dump instead of yaml.dump
   import sick

   original_model = sick.Model("model.yaml")
   cached_configuration = original_model.cache("grid_points.pickle", "fluxes.memmap",
       dispersion_filenames={"blue": "dispersion.memmap"},
       wavelengths=None,
       smoothing_kernels=None,
       sampling_rate=None)
   
   # OK, now let's save the new configuration
   with open("model.yaml", "w+") as fp:
       yaml.dump(cached_configuration, fp)


Inference
---------

Now that you have a cached model file, you are ready to start doing some inference. The easiest way to do this is to use the ``sick`` command line function. Let's say you have some observed data in a filename named ``my-star.fits``. Just do::

    sick solve model.yaml my-star.fits

And watch the crank turn! Once it's finished you will see a number of plots that you can examine to ensure everything makes sense.

.. note::
   ``sick`` can read in one-dimensional spectra in ASCII or FITS format. Some unconventional FITS formats are also allowed (e.g., AAOmega spectra), and more formats can be added. `Email me <mailto:arc@ast.cam.ac.uk>`_ if you are having trouble reading in your data.


Inference takes too long!
^^^^^^^^^^^^^^^^^^^^^^^^^

The amount of time ``sick`` takes to run will depend on:

1. The number of pixels in the synthetic grid,
2. Whether you have chosen to optimise the parameters prior to MCMC (recommended) or not,
3. The number of walkers, and
4. The number of burn-in and sampling steps to perform. 

There may be circumstances where your inference is taking much longer than it should. In these scenarios you have a few options available to you. 

Optimise the model grid
"""""""""""""""""""""""

Consider whether you need every single model in your grid. Can you still do your science if you restricted the range of model parameters or wavelength region? If the model spectra are of a **much** higher spectral resolution than the observed spectra, you might consider pre-smoothing the spectra to have a spectral resolution *just* higher than the observed data. And you could re-sample the spectra to eliminate a number of pixel points. If you have far too many redundant model pixels then you are unnecessarily slowing down the interpolation routine, so this is where you can make some big speed gains. I would recommend creating multiple cached copies of a grid at different convolutions and sampling, perform inference on each, and examine the posterior distributions from using each model.

Add prior information (if you've got it)
""""""""""""""""""""""""""""""""""""""""

Do you have any prior information (things you knew before taking the spectra, e.g., photometry) about the source object? If so, you can enter it into the model configuration file like this::

    priors:
      # From photometry we know alpha is 4523 +/- 180
      alpha: normal(4523, 180)

      # Any reasonable person would think beta can only be between -1 and 1 for this object
      beta: uniform(-1, 1)

Which will greatly speed up all stages (random scattering, optimisation, inference). You might also want to consider the number of walkers, burn-in steps and sampling steps you're performing.


Post-Processing
---------------

Unless the ``--no-plots`` flag is given, ``sick`` will produce a number of publication-quality figures after every inference. These are:

* A `triangle.py <https://github.com/dfm/triangle.py>`_ (written by `Dan Foreman-Mackey <http://dan.iel.fm/>`_) plot showing the posterior distributions of the astrophysical model parameters (see :py:func:`sick.plot.corner`)

* A projection of the maximum-likelihood spectrum on the data, as well as 100 random draws from the posterior distribution function (see :py:func:`sick.plot.projection`).

* The walker values at each MCMC step (see :py:func:`sick.plot.chains`).


By default ``sick`` will also save a number of other pertinent data. The final model state (a ``*.state`` file) is saved, allowing you to resume the analysis later on. The model configuration, maximum-likelihood parameters and credible uncertainties for each are stored in a JSON-formatted file. Samples from the posterior distribution are kept in a FITS table. See the help guide (``sick solve -h``) to disable these outputs.


Aggregating Results
-------------------

If you've analysed many (:math:`>1`) objects ``sick`` allows you to easily aggregate the results. The most relevant results for each inference are stored in JSON files, which you can join together into a single (FITS) table with the command::

    sick aggregate combined_table.fits *.json

Which will load in `TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_ and any other similar program.
 


Next steps
^^^^^^^^^^

Now that you have an overview of how everything works, why not check out these examples:

* Perform a self-consistent inference test with faux data

* Infer the stellar parameters of the Sun using a GIRAFFE/FLAMES twilight spectrum

* Analyse the SEGUE calibration star sample 



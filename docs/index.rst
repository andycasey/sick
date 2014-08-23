.. *sick* documentation master file, created by
   sphinx-quickstart on Thu Jul 17 10:44:35 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*sick*, the spectroscopic inference crank
=========================================

*sick* is a probabilstic code that allows you to approximate observed spectral data and infer properties of astrophysical sources. It can be used
on any kind of astrophysical object so long as you have a pre-computed grid of model spectra.

For example if you wanted to infer stellar parameters (effective temperature :math:`T_{\rm eff}`, surface gravity :math:`\log{g}`, metallicity :math:`[{\rm Fe/H}]`, 
alpha enhancement :math:`[\alpha/{\rm Fe}]`) from a SDSS spectrum (``J210495-2323434.fits``), once you've `installed sick <install.html>`_ it's as easy as doing::

    sick download AMBRE-model-grid 
    sick solve AMBRE-SDSS-example.yaml J210495-232343.fits

There is a getting started guide to explain how everything works, a number of examples using real data, a step-by-step tutorial on how to efficiently
cache your models and set up the problem, and an article explaining the software implementation. If something's not clear then please `create an issue
<http://github.com/andycasey/sick/issues/new>`_ or `email me <mailto:arc@ast.cam.ac.uk>`_.


User's Guide
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   install
   getting-started
   model-configuration-reference
   examples/
   api
   contributing
   attribution
   faq

Attribution
^^^^^^^^^^^

If you found this code useful in your research, please `let me know <mailto:arc@ast.cam.ac.uk>`_ and cite this paper::

    @BIBTEX


===================================================
SCOPE: Spectral Comparison and Parameter Evaluation
===================================================

[![Build Status](https://travis-ci.org/andycasey/a-new-scope.png?branch=master)](https://travis-ci.org/andycasey/a-new-scope) [![PyPi download count image](https://pypip.in/d/scope/badge.png)](https://pypi.python.org/pypi/scope/)

:Info: See the `GitHub repository <http://github.com/andycasey/a-new-scope/tree/master>`_ for the latest source
:Author: `Andy Casey <arc@ast.cam.ac.uk>`_ (arc@ast.cam.ac.uk)
:License: `Academic License <http://github.com/dfm/license>`_: This project includes academic-research code and documents under development. You would be a fool to run any of the code. Any use of the content requires citation.

Principle
=========
Most (if not all) attempts to determine stellar parameters from a grid of
spectra separate out the normalisation, radial velocity, and synthetic
smoothing components. In reality these are all linked, and the reliability
of stellar parameters will be affected by any uncertainty in these components.
You should have a single mathematical model that can incorporate all of these convolutions. 
That's what SCOPE does. It's flexible enough for use on any type of spectra.

Installation
============

``pip install scope`` (or [if you must](https://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install), use ``easy_install scope``)

Usage
=====
Check a configuration file:
``scope check config.yml``

Sometimes a configuration file will be set up with particular spectra in
mind, and will have wavelength-specific settings. You can check a config
file with a spectrum by doing:
``scope check my_spectrum.fits --with config.yml``

If it all checks out, you can analyse that spectrum:
``scope analyse my_spectrum.fits --with config.yml``

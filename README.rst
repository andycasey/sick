=========================
Star Wars IV: A New Scope
=========================

I'm fucking hilarious.

:Info: See the `GitHub repository <http://github.com/andycasey/a-new-scope/tree/master>`_ for the latest source
:Author: `Andy Casey <acasey@mso.anu.edu.au>`_ (acasey@mso.anu.edu.au)
:License: `Academic License <http://github.com/dfm/license>`_: This project includes academic-research code and documents under development. You would be a fool to run any of the code. Any use of the content requires citation.

Idea
====
Most (if not all) attempts to determine stellar parameters from a grid of
spectra separate out the normalisation and radial velocity components. In
reality these are all linked. You should have a single mathematical model
that can incorporate all of these convolutions. That's what this does.

Status
======
.. image:: https://travis-ci.org/andycasey/a-new-scope.png?branch=master :target: https://travis-ci.org/andycasey/a-new-scope

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

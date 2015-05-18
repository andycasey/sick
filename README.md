*sick*, the spectroscopic inference crank
------

[![Build Status](http://img.shields.io/travis/andycasey/sick.svg)](https://travis-ci.org/andycasey/sick) [![PyPi download count image](http://img.shields.io/pypi/dm/sick.svg)](https://pypi.python.org/pypi/sick/)

``sick`` is a generalised package for inferring astrophysical parameters from noisy observed spectra. Phenomena that can alter the data (e.g., redshift, continuum, instrumental broadening, outlier pixels) are modelled and simultaneously inferred with the astrophysical parameters of interest. This package is best-suited for situations where a grid of model spectra already exists, and you would like to infer model parameters given some data.

Installation
------------
You can install ``sick`` and all of its dependencies with the following one-liner:

``pip install sick --user``

(Or, [if you must](https://stackoverflow.com/questions/3220404/why-use-pip-over-easy-install), you can use ``easy_install`` instead of ``pip``)


Documentation
-------------
Guides, detailed examples, and documentation for ``sick`` can be found [here](http://astrowizici.st/sick/).


Attribution
-----------
Please cite ``Casey (2015, submitted)`` if you find this code useful in your research. The BibTeX entry for the paper is:

    @article{sick,
       author = {{Casey}, A.~R.},
        title = {sick: the spectroscopic inference crank},
      journal = {submitted},
         year = 2015,
    }
    
This code relies on the excellent [``emcee``](https://github.com/dfm/emcee) package, which you should also cite. If you use any pre-cached models available through the ``sick download`` command then please ensure you cite the appropriate works that originally published the spectra.

Bug reports/feature requests
----------------------------
Please [create an issue](https://github.com/andycasey/sick/issues/new) and be as descriptive as possible.

Contributing
------------
Contributions are warmly welcomed. To contribute to the code base please [fork this repository](https://github.com/andycasey/sick/fork), commit your changes, and create a pull request. 

License
-------
Copyright 2015, Andrew R. Casey. ``sick`` is free software released under the MIT License. See the ``LICENSE`` file for more details.


.. Frequently Asked Questions page 


==========================
Frequently Asked Questions
==========================

Here is a list of frequently asked questions (with answers), or questions that I might have if I were a new user.

.. rubric:: **Do all the model spectra need to have the same wavelength or frequency points?**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Yes. The multi-dimensional interpolation would be (at least) an order of magnitude more difficult if the model spectra were allowed to have different spectral dispersions. If you have model spectra that all have different wavelength/frequency points -- and you can't re-compute them to have the same dispersion -- then you will need to interpolate each of the model spectra to have the same dispersion.


.. rubric:: **What if I have a question that's not on this page?**
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

First see if you can find the answer in the documentation (use the search bar!) or in the `paper describing sick <arxiv.org>`_. If that doesn't help then please `open an issue <github.com/andycasey/sick/issues/new>`_ or `drop me a line <mailto:andy@astrowizici.st>`_.

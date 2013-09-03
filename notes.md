
Checking spectra
--------
- [x] Assume anything being given to ``scope`` at the moment is all part of the same star.
- [x] Check wavelengths of spectra and associate a standard spectrum to each observed spectrum
- [x] Check observed wavelengths dont overlap
- [x] Check standard arms don't overlap
- [x] Check coverage: Do standard arms cover full extend of observed spectra
- [x] Check observed pixel size is larger than synthetic arms in each
- [x] Put everything in blue -> red order

Checking standards
------------------
- [x] Run the filename matcher for each filename and ensure we get back dictionaries for each object
- [x] If there are multiple arms, check that a blue exists for red, etc


Checking model
--------------



Analysis
--------
- Do checks

- Radial velocity check against some synthetic template --> put at "rest"

- Initialise priors
- Start minimizing
   + normalises spectra with inputs
   + interpolates to synthetic flux
   + smoothes synthetic flux
   + shifts observed spectrum by v_rad
   + intepolates synthetic flux onto observed dispersion map
   + applies any masks
   + calculates chi^2 with either uncertainty in 1D spec, or sqrt(n)

- returns best parameters and map of inputs->chi^2
- calculates synthetic spectra for those params and saves it


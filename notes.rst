
Checking spectra
--------
[ ] Assume anything being given to ``scope`` at the moment is all part of the same star.
[ ] Check wavelengths of spectra and associate a standard spectrum to each observed spectrum
[ ] Check observed wavelengths dont overlap
[ ] Check standard arms don't overlap
[ ] Check coverage: Do standard arms cover full extend of observed spectra
[ ] Check observed pixel size is larger than synthetic arms in each
[ ] Put everything in blue -> red order

Checking standards
------------------
[ ] Run the filename matcher for each filename and ensure we get back dictionaries for each object
[ ] If there are multiple arms, check that a blue exists for red, etc


Checking model
--------------
normalise_observed
 - blue:
   - order:
   - knots:
 - red:

doppler_correct_observed:
 - red:

smooth_synthetic:
 - type: gaussian
 - kernel: free


Priors
------
smooth_synthetic
 - kernel: 0.10




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


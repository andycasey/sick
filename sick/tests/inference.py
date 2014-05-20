# coding: utf-8

""" Test internal consistency in SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@ast.cam.ac.uk>"

import numpy as np
import unittest


import scope

test_model_filename = "model.yaml"

#def setup(self):
#	""" Create a fake object """

model = scope.Model(test_model_filename)

# Map fake apertures
model._mapped_apertures = model.dispersion.keys()

spectra = None
while spectra is None:

	# Create a fake star
	truths = []
	for dimension in model.dimensions:
		if dimension in model.grid_boundaries:
			truths.append(np.random.uniform(*model.grid_boundaries[dimension]))

		elif dimension.startswith("doppler_shift."):
			truths.append(np.random.normal(0, 100)) # km/s

		elif dimension.startswith("smooth_model_flux."):
			truths.append(abs(np.random.normal(0, 1)))

		elif dimension.startswith("jitter."):
			# We will add noise later.
			truths.append(np.random.uniform(-10, 1))
			continue

		elif dimension.startswith("normalise_observed."):
			# Could be anything, but we must force it to be overall positive
			coefficient = int(dimension.split(".")[2][1:])
			if coefficient == 0:
				truths.append(np.random.uniform(1, 10))
			else:
				truths.append(0)

	# Create our star
	truths_dict = dict(zip(model.dimensions, truths))
	spectra = model.model_spectra(**truths_dict)

# Add noise
for aperture, spectrum in zip(model.apertures, spectra):
	spectrum.uncertainty = np.sqrt(spectrum.flux) + 0.5 * np.random.rand(len(spectrum.flux))

	spectrum.flux += np.abs(np.exp(truths_dict["jitter.{0}".format(aperture)]) * spectrum.flux) * np.random.randn(len(spectrum.flux))
	spectrum.flux += spectrum.uncertainty * np.random.randn(len(spectrum.flux))

# Just for fun: Re-sample the spectra so that we throw away half of the information.
observations = []
for spectrum in spectra:
	spectrum.disp = spectrum.disp[::2]
	spectrum.flux = spectrum.flux[::2]
	spectrum.uncertainty = spectrum.uncertainty[::2]
	observations.append(spectrum)

#def test_inference(:
#	""" Infer the parameters of our fake object """
raise foo
posterior, sampler, additional_info = scope.solve(observations, model)


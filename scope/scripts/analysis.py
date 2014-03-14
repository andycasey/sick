#!/usr/bin/python

""" Script to run SCOPE from the command line """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import argparse
import logging
import os

from hashlib import md5
from json import dumps as json_dumps
from time import time

import scope

logger = logging.basicConfig(level=logging.INFO)

def main():

	parser = argparse.ArgumentParser(description="infer stellar parameters from spectra")
	parser.add_argument("model", type=str, help="YAML- or JSON-style model filename")
	parser.add_argument("spectra", nargs="+", help="filenames of spectra to analyse")
	parser.add_argument("-o", "--output-dir", dest="output_dir", nargs="?", help="directory for output files",
		type=str, default=os.getcwd())
	parser.add_argument("--no-plots", dest="plotting", action="store_false", default=True)

	args = parser.parse_args()


	all_spectra = [scope.Spectrum.load(filename) for filename in args.spectra]

	# Are there multiple spectra in each aperture?
	if isinstance(all_spectra[0], list):
		# If so, they should all have the same length (e.g. same number of stars)
		if len(set(map(len, all_spectra))) > 1:
			raise IOError("filenames contain different number of spectra")

		# OK, they have the same length. They are probably apertures of the same
		# stars. Let's join them properly
		sorted_spectra = []
		num_stars, num_apertures = len(all_spectra[0]), len(all_spectra)
		for i in xrange(num_stars):
			sorted_spectra.append([all_spectra[j][i] for j in xrange(num_apertures)])

		all_spectra = sorted_spectra

	else:
		all_spectra = [all_spectra]

	# Define headers that we want in the results filename 
	default_headers = ("RA", "DEC", "COMMENT", "ELAPSED", "FIBRE_NUM", "LAT_OBS", "LONG_OBS",
		"MAGNITUDE","NAME", "OBJECT", "RO_GAIN", "RO_NOISE", "UTDATE", "UTEND", "UTSTART", )
	default_metadata = {
		"model": "",
		"filenames": ", ".join(args.spectra)
	}

	# For each spectra, analyse
	all_results = []
	for i, spectra in enumerate(all_spectra):

		# Create metadata and put header information in
		metadata = {}
		bluest_spectrum = spectra[0] if isinstance(spectra, list) else spectra
		for header in default_headers:
			if header not in bluest_spectrum.headers: continue
			metadata[header] = bluest_spectrum.headers[header]

		# Set defaults for metadata
		metadata.update(default_metadata)

		try:
			t_init = time()
			posteriors, sampler, model, mean_acceptance_fractions = scope.solve(spectra, args.model)

		except:
			logger.exception("Failed to analyse #{0}:".format(i))

			# Update the metadata with the model information, as well as NaN's for posteriors
			metadata.update({
				"model": md5(json_dumps(model.configuration).encode("utf-8")).hexdigest(),
				"time_elapsed": time() - t_init
			})
			for dimension in model.dimensions:
				metadata[dimension] = np.nan
				metadata["u_maxabs_{0}".format(dimension)] = np.nan
				metadata["u_pos{0}".format(dimension)] = np.nan
				metadata["u_neg_{0}".format(dimension)] = np.nan

		else:
			# Save information related to the analysis
			metadata.update({
				"model": md5(json_dumps(model.configuration).encode("utf-8")).hexdigest(),
				"time_elapsed": time() - t_init
			})

			# Update results with the posteriors
			for dimension, (posterior_value, pos_uncertainty, neg_uncertainty) in posteriors.iteritems():
				metadata.update({
					dimension: posterior_value,
					"u_maxabs_{0}".format(dimension): max(abs([neg_uncertainty, pos_uncertainty])),
					"u_pos_{0}".format(dimension): pos_uncertainty,
					"u_neg_{0}".format(dimension): neg_uncertainty
				})

			# Save the sampler chain
			blobs = np.array(sampler.chain)
			blobs = blobs.reshape((-1, 1 + len(model.dimensions)))
			chain = np.core.records.fromarrays(blobs.T,
				names=model.dimensions + ["ln L"],
				formats=["f8"] * (1 + len(model.dimensions)))

			with open(os.path.join(args.output_dir, "{0}.chain".format(i)), "wb") as fp:
				pickle.dump(chain, fp)

			# Plot results
			if not args.plotting: continue

		finally:
			# Save the metadata to the results list
			all_results.append(metadata)

			# Save all results thus far to a fits file
			raise a

if __name__ == "__main__":
	main()
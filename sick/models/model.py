#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Base model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
from time import time
from collections import OrderedDict

import numpy as np
import emcee
from astropy.constants import c as speed_of_light

import generate
from base import BaseModel
from .. import (inference, optimise as op, specutils)


logger = logging.getLogger("sick")

class Model(BaseModel):

    def estimate(self, data, full_output=False, **kwargs):
        """
        Estimate the model parameters, given the data.
        """

        # Number of model comparisons can be specified in the configuration.
        num_model_comparisons = self._configuration.get("estimate", {
                "num_model_comparisons": self.grid_points.size
            })["num_model_comparisons"]
        # If it's a fraction, we need to convert that to an integer.
        if 1 > num_model_comparisons > 0:
            num_model_comparisons *= self.grid_points.size

        # If the num_model_comparison is provided as a keyword argument, use it.
        num_model_comparisons = kwargs.pop("num_model_comparisons",
            int(num_model_comparisons))

        logger.debug("Number of model comparisons to make for initial estimate:"
            " {0}".format(num_model_comparisons))
        
        # Match the data to the model channels.
        matched_channels, missing_channels, ignore_parameters \
            = self._match_channels_to_data(data)

        logger.debug("Matched channels: {0}, missing channels: {1}, ignore "
            "parameters: {2}".format(matched_channels, missing_channels,
                ignore_parameters))

        # Load the intensities
        s = self.grid_points.size/num_model_comparisons # step size
        grid_points = self.grid_points[::s]
        intensities = np.memmap(
            self._configuration["model_grid"]["intensities"], dtype="float32",
            mode="r", shape=(self.grid_points.size, self.wavelengths.size))[::s]

        # Which matched, data channel has the highest S/N?
        # (This channel will be used to estimate astrophysical parameters)
        data, pixels_affected = self._apply_data_mask(data)
        median_snr = dict(zip(matched_channels,
            [np.nanmedian(spec.flux/(spec.variance**0.5)) for spec in data]))
        median_snr.pop(None, None) # Remove unmatched data spectra

        ccf_channel = self._configuration.get("settings", {}).get("ccf_channel",
            max(median_snr, key=median_snr.get))
        logger.debug("Channel with peak SNR is {0}".format(ccf_channel))

        # Are there *any* continuum parameters in any matched channel?
        any_continuum_parameters = any(map(lambda s: s.startswith("continuum_"),
            set(self.parameters).difference(ignore_parameters)))

        # [TODO]: Model mask.

        theta = {} # Dictionary for the estimated model parameters.
        best_grid_index = None
        c = speed_of_light.to("km/s").value
        for matched_channel, spectrum in zip(matched_channels, data):
            if matched_channel is None: continue

            # Do we need todo cross-correlation for this channel?
            # We do if there are redshift parameters for this channel,
            # or if there is a global redshift or global continuum parameters
            #   and this channel is the highest S/N.
            if "z_{}".format(matched_channel) in self.parameters \
            or ((any_continuum_parameters or "z" in self.parameters) \
            and matched_channel == ccf_channel):

                # Get the continuum degree for this channel.
                continuum_degree = self._configuration["model"].get("continuum",
                    { matched_channel: -1 })[matched_channel]

                logger.debug("Perfoming CCF on {0} channel with a continuum "
                    "degree of {1}".format(matched_channel, continuum_degree))

                # Get model wavelength indices that match the data.
                idx = np.clip(self.wavelengths[self._model_mask()].searchsorted(
                    [spectrum.disp[0], spectrum.disp[-1]]),
                    0, self.wavelengths.size)

                # Do the cross-correlation for this channel.
                # TODO: apply CCF mask.

                v, v_err, R = spectrum.cross_correlate(
                    (self.wavelengths[self._model_mask()][idx[0]:idx[1]],
                    intensities[:, self._model_mask()][:, idx[0]:idx[1]]),
                    continuum_degree=continuum_degree)

                # Apply limits:
                #lower, upper = -500, 500
                #R[~((upper > v) * (v > lower))] = np.nan

                # Identify the best point by the CCF peak.
                best = np.nanargmax(R)

                # Now, why did we do CCF in this channel? Which model parameters
                # should be updated?
                if "z_{}".format(matched_channel) in self.parameters:
                    theta["z_{}".format(matched_channel)] = v[best] / c
                elif "z" in self.parameters: 
                    # If there is a global redshift, update it.
                    theta["z"] = v[best] / c

                    # Continuum parameters will be updated later, so that each
                    # channel is checked to see if it has the highest S/N,
                    # otherwise we might be trying to calculate continuum
                    # parameters when we haven't done CCF on the highest S/N
                    # spectra yet.

                if matched_channel == ccf_channel:
                    # Update astrophysical parameters.
                    theta.update(dict(zip(grid_points.dtype.names,
                        grid_points[best])))
                    best_grid_index = best

        # If there are continuum parameters, calculate them from the best point.
        if any_continuum_parameters:
            for matched_channel, spectrum in zip(matched_channels, data):
                if matched_channel is None: continue

                # The template spectra at the best point needs to be
                # redshifted to the data, and then continuum coefficients
                # calculated from that.

                # Get the continuum degree for this channel.
                continuum_degree = self._configuration["model"].get("continuum",
                    { matched_channel: -1 })[matched_channel]

            
                # Get model wavelength indices that match the data.
                idx = np.clip(self.wavelengths.searchsorted(
                    [spectrum.disp[0], spectrum.disp[-1]]) + [0, 1],
                    0, self.wavelengths.size)

                # Redshift and bin the spectrum.
                z = theta.get("z_{}".format(matched_channel), theta.get("z", 0))

                best_intensities \
                    = np.copy(intensities[best_grid_index, idx[0]:idx[1]]).flatten()

                # Apply model mask.
                model_mask = self._model_mask(self.wavelengths[idx[0]:idx[1]])
                best_intensities[~model_mask] = np.nan

                best_intensities = best_intensities * specutils.sample.resample(
                    self.wavelengths[idx[0]:idx[1]] * (1 + z), spectrum.disp)
                
                # Calculate the continuum coefficients for this channel.
                continuum = spectrum.flux/best_intensities
                finite = np.isfinite(continuum)

                coefficients = np.polyfit(
                    spectrum.disp[finite], continuum[finite], continuum_degree,
                    w=spectrum.ivariance[finite])

                # They go into theta backwards. such that coefficients[-1] is
                # continuum_{name}_0
                theta.update(dict(zip(
                    ["continuum_{0}_{1}".format(matched_channel, i) \
                        for i in range(continuum_degree + 1)],
                    coefficients[::-1]
                )))

        # Remaining parameters could be: resolving power, outlier pixels,
        # underestimated variance.
        remaining_parameters = set(self.parameters)\
            .difference(ignore_parameters)\
            .difference(theta)

        if remaining_parameters:
            logger.debug("Remaining parameters to estimate: {0}. For these we "
                "will just assume reasonable initial values.".format(
                remaining_parameters))

            for parameter in remaining_parameters:
                if parameter == "resolution" \
                or parameter.startswith("resolution_"):

                    theta.update({ parameter: np.inf })

                elif parameter == "f" or parameter.startswith("f_"):
                    theta.update({ parameter: -10.0 }) # Not overestimated.

                elif parameter in ("Po", "Vo"):
                    theta.update({
                        "Po": 0.01, # 1% outlier pixels.
                        "Vo": max([np.nanmedian(s.variance) for s in data]),
                    })

        # Having full_output = True means return the best spectra estimate.
        if full_output:

            # Create model fluxes and calculate some metric.
            __intensities = np.copy(intensities[best_grid_index])

            # Apply model masks.
            __intensities[~self._model_mask()] = np.nan

            chi_sq, dof, model_fluxes = self._chi_sq(theta, data,
                __intensities=__intensities, __no_precomputed_binning=True)
            del intensities

            return (theta, chi_sq, dof, model_fluxes)

        # Delete the reference to intensities
        del intensities
        return theta


    def _initial_proposal_distribution(self, parameters, theta, size,
        default_std=1e-4):
        """
        Generate an initial proposal distribution around the point theta.
        """

        missing_parameters = set(parameters).difference(theta)
        if missing_parameters:
            raise ValueError("cannot create initial proposal distribution "\
                "because the following parameters are missing: {}".format(
                    ", ".join(missing_parameters)))

        std = np.ones(len(parameters), dtype=float)
        
        initial_proposal_stds \
            = self._configuration.get("initial_proposal_stds", {})

        p0 = np.array([theta[p] for p in parameters])
        std = np.array(map(float, [initial_proposal_stds.get(p, default_std) \
            for p in parameters]))

        return np.vstack([p0 + std * np.random.normal(size=len(p0)) \
            for i in range(size)])


    def _chi_sq(self, theta, data, **kwargs):

        chi_sq, dof = 0, -1
        model_fluxes = self(theta, data, **kwargs)

        for spectrum, model_flux in zip(data, model_fluxes):
            chi_sqi = (spectrum.flux - model_flux)**2 / spectrum.variance
            finite = np.isfinite(chi_sqi)

            chi_sq += chi_sqi[finite].sum()
            dof += finite.sum()

        return (chi_sq, dof, model_fluxes)
        

    def infer(self, data, initial_proposal=None, walkers=100, burn=5000,
        sample=5000, a=2.0, threads=1, full_output=False, **kwargs):
        """
        Infer the model parameters, given the data.
        """

        # Apply data masks now so we don't have to do it on the fly.
        data, pixels_affected = self._apply_data_mask(data)

        # Any channels / parameters to ignore?
        matched_channels, missing_channels, ignore_parameters \
            = self._match_channels_to_data(data)
        parameters = [p for p in self.parameters if p not in ignore_parameters]
        #parameters = list(set(self.parameters).difference(ignore_parameters))

        logger.debug("Inferring {0} parameters: {1}".format(len(parameters),
            ", ".join(parameters)))

        if walkers % 2 > 0 or walkers < 2 * len(parameters):
            raise ValueError("the number of walkers must be an even number and "
                "be at least twice the number of model parameters")

        _be_positive = {
            "burn": burn,
            "sample": sample,
            "threads": threads
        }
        for k, v in _be_positive.items():
            if v < 1:
                raise ValueError("{} must be a positive integer".format(k))

        # Get the inference keyword arguments.
        infer_kwargs = self._configuration.get("infer", {}).copy()
        infer_kwargs.update(kwargs)

        """
        fixed = infer_kwargs.pop("fixed", {})
        if fixed:
            # Remove non-parameters from the 'fixed' keywords.
            keys = set(fixed).intersection(parameters)
            # If the 'fixed' value is provided, use that. Otherwise if it is
            # None then use the initial_theta value.
            fixed = dict(zip(keys, 
                [(fixed[k], initial_theta.get(k, None))[fixed[k] is None] \
                    for k in keys]))

            logger.info("Fixing keyword arguments (these will not be inferred)"\
                ": {}".format(fixed))

        # Remove fixed parameters from the parameters to be optimised
        #parameters = list(set(parameters).difference(fixed))
        parameters = [p for p in parameters if p not in fixed]
        """


        # Initial proposal could be:
        #   - an array (N_walkers, N_dimensions)
        #   - a dictionary containing key/value pairs for the dimensions
        #   - None
        if initial_proposal is None:
            initial_proposal = self.estimate(data)

        if isinstance(initial_proposal, dict):
            initial_proposal = self._initial_proposal_distribution(
                parameters, initial_proposal, walkers)

        elif isinstance(initial_proposal, np.ndarray):
            initial_proposal = np.atleast_2d(initial_proposal)
            if initial_proposal.shape != (walkers, len(parameters)):
                raise ValueError("initial proposal must be an array of shape "\
                    "(N_parameters, N_walkers) ({0}, {1})".format(walkers,
                        len(parameters)))

        """
        # Pre-create binning matrices if redshift and resolution are not to be
        # optimised.
        create_binning_matrices = not any([p.startswith("resolution_") or \
            p.startswith("z_") or p in ("z", "resolution") for p in parameters])

        if create_binning_matrices:
            logger.debug("Creating rebinning matrices prior to inference")

            # Create binning matrices for each channel.
            matrices = []
            for channel, spectrum in zip(matched_channels, data):
                if channel is None: 
                    matrices.append(None)
                    continue

                z = 0 #fixed.get("z", fixed.get("z_{}".format(channel), 0))
                
                resolution = None #fixed.get("resolution", fixed.get("resolution_{}"\
                #    .format(channel), None))

                # TODO: this may be unnecessarily expensive when two channels
                #       have wavelengths very far from each other.
                #       (consider splicing to the wavelengths for this channel)
                if resolution is None:
                    matrices.append(specutils.sample.resample(
                        generate.wavelengths[-1] * (1. + z),
                        spectrum.disp))

                else:
                    # TODO: get old_resolution from metadata
                    matrices.append(specutils.sample.resample_and_convolve(
                        generate.wavelengths[-1] * (1. + z),
                        spectrum.disp, new_resolution=resolution))

            # Make the binning matrices accessible globally.
            generate.binning_matrices.append(matrices)
        """
        logger.info("Creating box factories...")
        calculate_binning_matrices = not self._configuration.get("settings",
            {}).get("fast_binning", True)
        if calculate_binning_matrices:
            matrix_factories = []
            for channel, spectrum in zip(matched_channels, data):
                if channel is None:
                    matrix_factories.append(None)
                    continue

                # Should it be a BlurryBoxFactory or a BoxFactory?
                klass = specutils.sample._BlurryBoxFactory if any([p.startswith(
                    "resolution_") or p == "resolution" for p in parameters]) else \
                    specutils.sample._BoxFactory

                # TODO: provide 'old_resolution' metadata.
                matrix_factories.append(
                    klass(spectrum.disp, generate.wavelengths[-1]))

            # Make the binning factories globally accessible.
            generate.binning_matrices.append(matrix_factories)

        # Check for non-standard proposal scales.
        if a != 2.0:
            logger.warn("Using proposal scale of {0:.2f}".format(a))

        # Create the sampler.
        logger.info("Creating sampler with {0} walkers and {1} threads".format(
            walkers, threads))
        debug = kwargs.get("debug", False)
        sampler = emcee.EnsembleSampler(walkers, len(self.parameters),
            inference.ln_probability, args=(parameters, self, data, debug), a=a,
            threads=threads)
        
        # Burn in.
        sampler, burn_acceptance_fractions, pos, lnprob, rstate, burn_elapsed \
            = self._sample(sampler, initial_proposal, burn, descr="burn-in")

        # Save the chain and log probabilities before we reset the chain.
        burn_chains = sampler.chain
        burn_ln_probabilities = sampler.lnprobability

        # Reset the chain.
        logger.debug("Resetting chain...")
        sampler.reset()

        # Sampler.
        sampler, prod_acceptance_fractions, pos, lnprob, rstate, prod_elapsed \
            = self._sample(sampler, pos, sample, lnprob0=lnprob, rstate0=rstate, 
                descr="production")

        if sampler.pool:
            sampler.pool.close()
            sampler.pool.join()
        
        logger.info("Time elapsed for burn / production / total: {0:.1f} "
            "{1:.1f} {2:.1f}".format(burn_elapsed, prod_elapsed, burn_elapsed +
                prod_elapsed))

        # Stack burn and production information together.
        chains = np.hstack([burn_chains, sampler.chain])
        lnprobability = np.hstack([burn_ln_probabilities, sampler.lnprobability])
        acceptance_fractions \
            = np.hstack([burn_acceptance_fractions, prod_acceptance_fractions])

        chi_sq, dof, model_fluxes = self._chi_sq(dict(zip(parameters, 
            [np.percentile(chains[:, burn:, i], 50) 
                for i in range(len(parameters))])), data)

        # Convert velocity scales.
        symbol, scale, units = self._preferred_redshift_scale
        labels = [] + parameters
        scales = np.ones(len(parameters))
        if symbol != "z":
            for i, parameter in enumerate(parameters):
                if parameter == "z" or parameter.startswith("z_"):
                    chains[:, :, i] *= scale
                    scales[i] = scale
                    if "_" in parameter:
                        labels[i] = "_".join([symbol, parameter.split("_")[1:]])
                    else:
                        labels[i] = symbol
                    logger.debug("Scaled {0} (now {1}) to units of {2}".format(
                        parameter, labels[i], units))

        # Calculate MAP values and associated uncertainties.
        theta = OrderedDict()
        for i, label in enumerate(labels):
            l, c, u = np.percentile(chains[:, burn:, i], [16, 50, 84])
            theta[label] = (c, u-c, l-c)

        # Re-arrange the chains to be in the same order as the model parameters.
        indices = np.array([parameters.index(p) \
            for p in self.parameters if p in parameters])
        chains = chains[:, :, indices]

        # Remove any pre-calculated binning matrices.
        if calculate_binning_matrices:
            logger.debug("Removed pre-calculated binning matrices")
            generate.binning_matrices.pop(-1)

        if full_output:
            metadata = {
                "burn": burn,
                "walkers": walkers,
                "sample": sample,
                "parameters": labels,
                "scales": scales,
                "chi_sq": chi_sq,
                "dof": dof
            }
            return (theta, chains, lnprobability, acceptance_fractions, sampler,
                metadata)
        return theta


    def _sample(self, sampler, p0, iterations, descr=None, **kwargs):

        runtime_descr = "" if descr is None else " of {}".format(descr)
        mean_acceptance_fraction = np.zeros(iterations)

        t_init = time()
        for i, (pos, lnprob, rstate) \
        in enumerate(sampler.sample(p0, iterations=iterations, **kwargs)):
            mean_acceptance_fraction[i] = sampler.acceptance_fraction.mean()

            # Announce progress.
            logger.info(u"Sampler at step {0:.0f}{1} has a mean acceptance "
                "fraction of {2:.3f} and highest log probability was {3:.3e}"\
                .format(i + 1, runtime_descr, mean_acceptance_fraction[i],
                    sampler.lnprobability[:, i].max()))

            if mean_acceptance_fraction[i] in (0, 1):
                raise RuntimeError("mean acceptance fraction is {0:.0f}".format(
                    mean_acceptance_fraction[i]))

            """
            if i % 100 == 0 and i > 0:
                # Do autocorrelation time.
                try:
                    tau_eff = emcee.autocorr.integrated_time(
                        sampler.chain[:, :i, :].reshape(-1, p0.shape[1]))
                except ValueError:
                    logger.debug("Could not calculate integrated autocorrelation"
                        " times")

                else:
                    N = p0.shape[0] * (i + 1)
                    effective_samples = N / (2 * tau_eff)

                    print(effective_samples)
                    print("MINIMUM NUMBER OF EFFECTIVE SAMPLES {0:.0f}".format(
                        effective_samples.min()))
            """

        elapsed = time() - t_init
        logger.debug("Sampling{0} took {1:.1f} seconds".format(
            "" if not descr else " ({})".format(descr), elapsed))

        return (sampler, mean_acceptance_fraction, pos, lnprob, rstate, elapsed)



    def optimise(self, data, initial_theta=None, full_output=False, **kwargs):
        """
        Optimise the model parameters, given the data.
        """

        data = self._format_data(data)

        if initial_theta is None:
            initial_theta = self.estimate(data)

        # Which parameters will be optimised, and which will be fixed?
        matched_channels, missing_channels, ignore_parameters \
            = self._match_channels_to_data(data)
        #parameters = set(self.parameters).difference(ignore_parameters)
        parameters = [p for p in self.parameters if p not in ignore_parameters]

        # What model wavelength ranges will be required?
        wavelengths_required = []
        for channel, spectrum in zip(matched_channels, data):
            if channel is None: continue
            z = initial_theta.get("z",
                initial_theta.get("z_{}".format(channel), 0))
            wavelengths_required.append(
                [spectrum.disp[0] * (1 - z), spectrum.disp[-1] * (1 - z)])

        # Create the spectrum approximator/interpolator.
        # TODO: Allow rescale command for the approximator.
        closest_theta = [initial_theta[p] for p in self.grid_points.dtype.names]
        subset_bounds = self._initialise_approximator(
            closest_theta=closest_theta, 
            wavelengths_required=wavelengths_required)
        
        # Get the optimisation keyword arguments.
        op_kwargs = self._configuration.get("optimise", {}).copy()
        op_kwargs.update(kwargs)

        # Get fixed keywords.
        fixed = op_kwargs.pop("fixed", {})
        if fixed:
            # Remove non-parameters from the 'fixed' keywords.
            keys = set(fixed).intersection(parameters)
            # If the 'fixed' value is provided, use that. Otherwise if it is
            # None then use the initial_theta value.
            fixed = dict(zip(keys, 
                [(fixed[k], initial_theta.get(k, None))[fixed[k] is None] \
                    for k in keys]))

            logger.info("Fixing keyword arguments (these will not be optimised)"
                ": {}".format(fixed))

        # Remove fixed parameters from the parameters to be optimised
        #parameters = list(set(parameters).difference(fixed))
        parameters = [p for p in parameters if p not in fixed]

        # Translate input bounds.
        nbs = (None, None) # No boundaries.
        input_bounds = op_kwargs.pop("bounds", {})
        op_kwargs["bounds"] = [input_bounds.get(p, subset_bounds.get(p, nbs)) \
            for p in parameters]
        
        # Apply data masks now so we don't have to do it on the fly.
        masked_data, pixels_affected = self._apply_data_mask(data)

        # Pre-create binning matrix factories.
        calculate_binning_matrices = (not self._configuration.get("settings",
            {}).get("fast_binning", True)) \
            or any([p == "z" or p.startswith("z_") for p in fixed])
        if calculate_binning_matrices:
            logger.debug("Creating box factories...")
            matrix_factories = []
            for channel, spectrum in zip(matched_channels, data):
                if channel is None:
                    matrix_factories.append(None)
                    continue

                # Should it be a BlurryBoxFactory or a BoxFactory?
                klass = specutils.sample._BlurryBoxFactory if any([p.startswith(
                    "resolution_") or p == "resolution" for p in parameters]) else \
                    specutils.sample._BoxFactory

                # If z is fixed then adjust the wavelengths.
                #fixed_z = fixed.get("z", fixed.get("z_{}".format(channel), 0))
                # TODO: provide 'old_resolution' metadata.
                matrix_factories.append(
                    klass(spectrum.disp, generate.wavelengths[-1]))

            # Make the binning factories globally accessible.
            generate.binning_matrices.append(matrix_factories)

        logger.info("Optimising parameters: {0}".format(", ".join(parameters)))
        logger.info("Optimisation keywords: {0}".format(op_kwargs))

        # Create the objective function.
        debug = kwargs.get("debug", False)
        def nlp(theta):
            # Apply fixed keywords
            t_ = theta.copy()
            p_ = [] + parameters
            for parameter, value in fixed.iteritems():
                p_.append(parameter)
                t_ = np.append(theta, value)

            return -inference.ln_probability(t_, p_, self, data, debug)

        # Do the optimisation.
        p0 = np.array([initial_theta[p] for p in parameters])
        x_opt = op.minimise(nlp, p0, **op_kwargs)

        # Put the result into a usable form.
        x_opt_theta = OrderedDict(zip(parameters, x_opt))
        # TODO: MAKE SURE THE x_opt_theta IS IN THE SAME ORDER AS MODEL.pARAMETERS
        x_opt_theta.update(fixed)

        if full_output:
            # Create model fluxes and calculate some metric.
            chi_sq, dof, model_fluxes = self._chi_sq(x_opt_theta, data)

            # Remove any pre-calculated binning matrices.
            if calculate_binning_matrices:
                logger.debug("Removed pre-calculated binning matrices")
                generate.binning_matrices.pop(-1)

            return (x_opt_theta, chi_sq, dof, model_fluxes)

        # Remove any pre-calculated binning matrices.
        if calculate_binning_matrices:
            logger.debug("Removed pre-calculated binning matrices")
            generate.binning_matrices.pop(-1)

        return x_opt_theta


    def __call__(self, theta, data, debug=False, **kwargs):

        if not isinstance(data, (list, tuple)):
            data = [data]

        if not isinstance(theta, dict):
            theta = dict(zip(self.parameters, theta))

        if "__intensities" in kwargs:
            logger.debug("Using __intensities")
            model_wavelengths = self.wavelengths
            model_intensities = kwargs.pop("__intensities")
            model_variances = 0

        else:
            model_wavelengths, model_intensities, model_variances \
                = self._approximate_intensities(theta, data, debug=debug, **kwargs)

        model_fluxes = []
        model_flux_variances = []

        matched_channels, _, __ = self._match_channels_to_data(data)
        for i, (channel, spectrum) in enumerate(zip(matched_channels, data)):
            if channel is None:
                _ = np.nan * np.ones(spectrum.disp.size)
                model_fluxes.append(_)
                model_flux_variances.append(_)
                continue

            # Get the redshift and resolution.
            z = theta.get("z", theta.get("z_{}".format(channel), 0))
            resolution = theta.get("resolution", theta.get("resolution_{}"\
                .format(channel), None))

            if kwargs.get("__no_precomputed_binning", False):
                if resolution:
                    matrix = specutils.sample.resample_and_convolve(
                        self.wavelengths * (1 + z), spectrum.disp,
                        resolution)
                else:

                    matrix = specutils.sample.resample(
                        self.wavelengths * (1 + z), spectrum.disp)

                channel_fluxes = model_intensities * matrix
                channel_variance = model_variances * matrix


            else:
                t = time()
                # Get the pre-calculated rebinning matrix.
                try:
                    matrix = generate.binning_matrices[-1][i]

                except TypeError:
                    if resolution is None:
                        channel_fluxes = np.interp(spectrum.disp,
                            model_wavelengths * (1 + z), model_intensities,
                            left=np.nan, right=np.nan)
                        if model_variances != 0:
                            channel_variance = np.interp(spectrum.disp,
                                model_wavelengths * (1 + z), model_variances,
                                left=np.nan, right=np.nan)
                        else:
                            channel_variance = np.zeros_like(channel_fluxes)

                    else:
                        logger.exception("Could not find binning matrices")
                        raise

                else:
                    try:
                        # If it's a callable, provide all possible quantities
                        # and it will ignore those that it does not need.
                        matrix = matrix(resolution=resolution, z=-z)
                    except TypeError:
                        None

                    else:
                        channel_fluxes = model_intensities * matrix
                        channel_variance = model_variances * matrix

                #print(resolution, z, time() - t)

            # Apply continuum if it is present.
            i, coeffs = 0, []
            while theta.get("continuum_{0}_{1}".format(channel, i), None):
                coeffs.append(theta["continuum_{0}_{1}".format(channel, i)])
                i += 1

            if coeffs:
                channel_fluxes *= np.polyval(coeffs[::-1], spectrum.disp)

            model_fluxes.append(channel_fluxes)
            model_flux_variances.append(channel_variance)

        # TODO check channel fluxes are not zero at the edges.
        if kwargs.pop("full_output", False):
            return (model_fluxes, model_flux_variances, matched_channels)

        return model_fluxes

    # Functions that should be overwritten by subclasses..
    def _approximate_intensities(self, *args, **kwargs):
        raise NotImplementedError("this should be overwritten in a subclass")

    def _initalise_approximator(self, *args, **kwargs):
        raise NotImplementedError("this should be overwritten in a subclass")



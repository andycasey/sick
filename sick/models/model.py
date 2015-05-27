#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Base model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import curses
import logging
import sys
from time import time
from collections import OrderedDict

import numpy as np
import emcee
from astropy.constants import c as speed_of_light
from scipy.ndimage import gaussian_filter1d

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
        num_model_comparisons = self._configuration.get("estimate", {}).get(
            "num_model_comparisons", self.grid_points.size)
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
        t = time()
        s = self.grid_points.size/num_model_comparisons # step size
        grid_points = self.grid_points[::s]
        intensities = np.memmap(
            self._configuration["model_grid"]["intensities"], dtype="float32",
            mode="r", shape=(self.grid_points.size, self.wavelengths.size))[::s]
        logger.debug("Took {:.0f} seconds to load and slice intensities".format(
            time() - t))
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

        # [TODO]: CCF MASK
        # [TODO]: Don't require CCF if we have only continuum parameters.

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
                # get the points that are in the mask, and within the spectrum
                # limits

                # TODO: Make this CCF not model mask.
                idx = np.where(self._model_mask() \
                    * (self.wavelengths >= spectrum.disp[0]) \
                    * (spectrum.disp[-1] >= self.wavelengths))[0]

                v, v_err, R = spectrum.cross_correlate(
                    #(self.wavelengths[idx], intensities[:, idx]),
                    (self.wavelengths, intensities),
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

                    if parameter.startswith("resolution_"):
                        spectra = [data[matched_channels.index(
                            parameter.split("_")[1])]]
                    else:
                        spectra = [s for s in data if s is not None]

                    R = [s.disp.mean()/np.diff(s.disp).mean() for s in spectra]

                    # Assume oversampling rate of ~5.
                    theta.update({ parameter: np.median(R)/5.})

                elif parameter == "f" or parameter.startswith("f_"):
                    theta.update({ parameter: -10.0 }) # Not overestimated.

                elif parameter in ("Po", "Vo"):
                    theta.update({
                        "Po": 0.01, # 1% outlier pixels.
                        "Vo": np.mean([np.nanmedian(s.variance) for s in data]),
                    })

        logger.info("Initial estimate: {}".format(theta))
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

        # Create the spectrum approximator/interpolator.
        # TODO: Allow rescale command for the approximator.
        
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

        # Initial proposal could be:
        #   - an array (N_walkers, N_dimensions)
        #   - a dictionary containing key/value pairs for the dimensions
        #   - None
        if initial_proposal is None:
            initial_proposal = self.estimate(data)

        if isinstance(initial_proposal, dict):

            wavelengths_required = []
            for channel, spectrum in zip(matched_channels, data):
                if channel is None: continue
                z = initial_proposal.get("z",
                    initial_proposal.get("z_{}".format(channel), 0))
                wavelengths_required.append(
                    [spectrum.disp[0] * (1 - z), spectrum.disp[-1] * (1 - z)])

            closest_point = [initial_proposal[p] for p in self.grid_points.dtype.names]
            subset_bounds = self._initialise_approximator(
                closest_point=closest_point, 
                wavelengths_required=wavelengths_required, force=True, **kwargs)

            initial_proposal = self._initial_proposal_distribution(
                parameters, initial_proposal, walkers)

        elif isinstance(initial_proposal, np.ndarray):
            initial_proposal = np.atleast_2d(initial_proposal)
            if initial_proposal.shape != (walkers, len(parameters)):
                raise ValueError("initial proposal must be an array of shape "\
                    "(N_parameters, N_walkers) ({0}, {1})".format(walkers,
                        len(parameters)))

        # Prepare the convolution functions.
        self._create_convolution_functions(matched_channels, data, parameters)

        # Check for non-standard proposal scales.
        if a != 2.0:
            logger.warn("Using proposal scale of {0:.2f}".format(a))

        # Create the sampler.
        logger.info("Creating sampler with {0} walkers and {1} threads".format(
            walkers, threads))
        debug = kwargs.get("debug", False)
        sampler = emcee.EnsembleSampler(walkers, len(parameters),
            inference.ln_probability, a=a, args=(parameters, self, data, debug),
            kwargs={"matched_channels": matched_channels}, threads=threads)

        # Burn in.
        progress_bar = kwargs.get("__show_progress_bar", True)
        sampler, burn_acceptance_fractions, pos, lnprob, rstate, burn_elapsed \
            = self._sample(sampler, initial_proposal, burn, descr="burn-in",
                parameters=parameters, __show_progress_bar=progress_bar)

        # Save the chain and log probabilities before we reset the chain.
        burn_chains = sampler.chain
        burn_ln_probabilities = sampler.lnprobability

        # Reset the chain.
        logger.debug("Resetting chain...")
        sampler.reset()

        # Sampler.
        sampler, prod_acceptance_fractions, pos, lnprob, rstate, prod_elapsed \
            = self._sample(sampler, pos, sample, lnprob0=lnprob, rstate0=rstate, 
                descr="production", parameters=parameters,
                __show_progress_bar=progress_bar)
                
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

        # Remove the convolution functions.
        if not kwargs.get("__keep_convolution_functions", False):
            self._destroy_convolution_functions()

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

        progress_bar = kwargs.pop("__show_progress_bar", True)
        parameters = kwargs.pop("parameters", np.arange(sampler.chain.shape[2]))
        runtime_descr = "" if descr is None else " of {}".format(descr)
        mean_acceptance_fraction = np.zeros(iterations)

        increment = int(iterations / 100)

        if progress_bar:
            screen = curses.initscr()
            curses.noecho()
            curses.cbreak()

        t_init = time()
        for i, (pos, lnprob, rstate) \
        in enumerate(sampler.sample(p0, iterations=iterations, **kwargs)):
            mean_acceptance_fraction[i] = sampler.acceptance_fraction.mean()

            if progress_bar:

                screen.addstr(0, 0,
                    "\rSampler at step {0:.0f}{1} with a mean accept"\
                    "ance fraction of {2:.3f}, highest ln(P) was {3:.3e}\n"\
                    .format(i + 1, runtime_descr,
                        mean_acceptance_fraction[i],
                        sampler.lnprobability[:, i].max()))

                if (i % increment == 0):
                    
                    message = "[{done}{not_done}] {percent:3.0f}%".format(
                        done="=" * int(i / increment),
                        not_done=" " * int((iterations - i)/increment),
                        percent=100.*i/iterations)
                    screen.addstr(1, 0, message)

                    if i > 0:
                        for j, parameter in enumerate(parameters):
                            pcs = np.percentile(
                                sampler.chain[:, i-increment:i, j], [16, 50, 84])
                            message = "\t{0}: {1:.3f} ({2:+.3f}, {3:+.3f})".format(
                                parameter, pcs[1], pcs[2] - pcs[1], -pcs[1] + pcs[0])

                            if parameter == "z" or parameter[:2] == "z_":
                                pcs *= 299792.458
                                message += " [{0:.1f} ({1:+.1f}, {2:+.1f}) km/s]"\
                                    .format(pcs[1], pcs[2] - pcs[1], -pcs[1] + pcs[0])
                            
                            screen.addstr(j + 2, 0, message)

                screen.refresh()

            else:
                # Announce progress.
                logger.info("Sampler at step {0:.0f}{1} has a mean acceptance f"
                    "raction of {2:.3f} and highest ln probability was {3:.3e}"\
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
        
        curses.echo()
        curses.nocbreak()
        curses.endwin()

        elapsed = time() - t_init
        logger.debug("Sampling{0} took {1:.1f} seconds".format(
            "" if not descr else " ({})".format(descr), elapsed))

        return (sampler, mean_acceptance_fraction, pos, lnprob, rstate, elapsed)


    def _create_convolution_functions(self, matched_channels, data, free_parameters):
        """
        Pre-create binning matrix factories. The following options need to be
        followed on a per-matched channel basis.

        Options here are:

        1) If we have no redshift or resolution parameters to solve for, then
           we can just produce a matrix that will be multiplied in.
           Inputs: none
           Outputs: matrix

        If fast_binning is turned on:

        2) Provide the wavelengths, redshift, resolution (single value) and
           flux. Returns the convolved, interpolated flux.
           Inputs: flux, wavelengths (e.g., redshift), resolution
           Outputs: normalised flux.

        If fast_binning is turned off:

        3) If we *just* have redshift parameters to solve for, then we can
           produce a _BoxFactory that will take a redshift z and return
           the matrix
           [This function will have a LRU cacher and should be removed after]
           Inputs: redshift
           Outputs: matrix

        4) If we have redshift parameters and resolution parameters, then we
           can produce a _BlurryBoxFactory that will take a redshift z and
           Resolution (or resolution coefficients!) and return a binning matrix
           [This function will have a LRU cacher and should be removed after]
           Inputs: redshift, Resolution parameter(s)
           Outputs: matrix

        5) If we *just* have resolution parameters to solve for, then we 
           can produce a _BlurryBoxFactory as well, because by default z=0.
           [This function will have a LRU cacher and should be removed after]
           Inputs: resolution parameter(s)
           Outputs: matrix


        Because these options require different input/outputs, and the LRU
        cachers need to remain wrapped around simple functions, we may need a

        Consistent lambda function:
        lambda(obs_wavelength, obs_flux, z=0, *R)
        """

        fast_binning = self._configuration.get("settings", {}).get(
            "fast_binning", 1)

        logger.info("Creating convolution functions (fast_binning = {})".format(
            fast_binning))

        convolution_functions = []
        for channel, spectrum in zip(matched_channels, data):
            if channel is None:
                convolution_functions.append(None)
                continue

            # Any redshift or resolution parameters?
            redshift = "z" in free_parameters \
                or "z_{}".format(channel) in free_parameters
            resolution = "resolution_{}".format(channel) in free_parameters

            # Option 1.
            # Create static binning matrices for each channel.
            if not redshift and not resolution:
                logger.info("Creating static matrix for channel {0} because "\
                    "no redshift or resolution parameters were found".format(
                        channel))

                # Create the binning matrix based on the globally-scoped array
                # of wavelengths.
                matrix = specutils.sample.resample(generate.wavelengths[-1],
                    spectrum.disp)

                # Wrap in a lambda function to be consistent with other options.
                convolution_function = lambda w, f, *a: f * matrix

            else:

                # If fast_binning is turned on (default):
                if fast_binning:
                    
                    # Option 2: Convolve with single kernel & interpolate.
                    # [TODO] should w.mean()/R be squared?
                    # px_sigma ~= ((w.mean()/R) / 2.35482)/np.diff(w).mean()
                    # px_scale = ((w.mean()/R) / 2.35)/np.diff(w).mean()
                    # px_scale = R_scale / R

                    logger.info("Creating simple convolution & interpolating "\
                        "function for channel {0}".format(channel))

                    # [TODO] Account for the existing spectral resolution of the
                    # grid.
                    if resolution:
                        R_scale = spectrum.disp.mean() \
                            / (2.35482 * np.diff(spectrum.disp).mean())
                        """
                        convolution_function = lambda w, f, z, R, *a: \
                            np.interp(w, generate.wavelengths[-1] * (1 + z),
                                gaussian_filter1d(f, max(0, R_scale/R),
                                    mode="constant", cval=np.nan),
                                left=np.nan, right=np.nan)
                        """
                        def convolution_function(w, f, z, R, *a):
                            if R > 0:
                                _ = gaussian_filter1d(f, R_scale/R)
                            else:
                                _ = f
                            return np.interp(w, generate.wavelengths[-1] * (1 + z),
                                _, left=np.nan, right=np.nan)

                    else:
                        convolution_function = lambda w, f, z, *a: \
                            np.interp(w, generate.wavelengths[-1] * (1 + z), f,
                                left=np.nan, right=np.nan)

                else:
                    if redshift and not resolution:
                        # Option 3: Produce a _BoxFactory
                        logger.info("Producing a Box Factory for convolution "\
                            "in channel {}".format(channel))

                        matrix = specutils.sample._BoxFactory(
                            spectrum.disp, generate.wavelengths[-1])

                        # Wrap in a lambda function to be consistent.
                        convolution_function = lambda w, f, z, *a: f * matrix(z)

                    else:
                        # Could be redshift and resolution, or just resolution.
                        # Options 4 and 5: Produce a _BlurryBoxFactory
                        logger.info("Producing a Blurry Box Factory for "\
                            "convolution in channel {}".format(channel))

                        matrix = specutils.sample._BlurryBoxFactory(
                            spectrum.disp, generate.wavelengths[-1])

                        # Wrap in a lambda function to be consistent.
                        convolution_function \
                            = lambda w, f, z, R, *a: f * matrix(R, z)

            # Append this channel's convolution function.
            convolution_functions.append(convolution_function)

        # Put the convolution functions into the global scope.
        generate.binning_matrices.append(convolution_functions)
        return True


    def _destroy_convolution_functions(self):
        logger.info("Removing run-time convolution functions.")
        _ = generate.binning_matrices.pop(-1)
        return True


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
        closest_point = [initial_theta[p] for p in self.grid_points.dtype.names]
        subset_bounds = self._initialise_approximator(
            closest_point=closest_point, 
            wavelengths_required=wavelengths_required, **kwargs)
        
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

        # Prepare the convolution functions.
        self._create_convolution_functions(matched_channels, data, parameters)

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

            return -inference.ln_probability(t_, p_, self, data, debug,
                matched_channels=matched_channels)

        # Do the optimisation.
        p0 = np.array([initial_theta[p] for p in parameters])
        x_opt = op.minimise(nlp, p0, **op_kwargs)

        # Put the result into a usable form.
        x_opt_theta_unscaled = OrderedDict(zip(parameters, x_opt))
        # TODO: MAKE SURE THE x_opt_theta IS IN THE SAME ORDER AS MODEL.pARAMETERS
        x_opt_theta_unscaled.update(fixed)

        if hasattr(self, "_cannon_offsets"):
            x_opt[:self._cannon_offsets.size] += self._cannon_offsets
        x_opt_theta = OrderedDict(zip(parameters, x_opt))
        x_opt_theta.update(fixed)

        if full_output:
            # Create model fluxes and calculate some metric.
            chi_sq, dof, model_fluxes = self._chi_sq(x_opt_theta_unscaled, data)

            # Remove any prepared convolution functions.
            self._destroy_convolution_functions()
    
            return (x_opt_theta, chi_sq, dof, model_fluxes)

        # Remove any prepared convolution functions.
        self._destroy_convolution_functions()

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
            model_variances = np.zeros_like(model_wavelengths)

        else:
            model_wavelengths, model_intensities, model_variances \
                = self._approximate_intensities(theta, data, debug=debug, **kwargs)

        continua = []
        model_fluxes = []
        model_flux_variances = []

        matched_channels = kwargs.get("matched_channels", None)
        if matched_channels is None:
            matched_channels, _, __ = self._match_channels_to_data(data)
        
        no_precomputed_binning = kwargs.get("__no_precomputed_binning", False)
        for i, (channel, spectrum) in enumerate(zip(matched_channels, data)):
            if channel is None:
                _ = np.nan * np.ones(spectrum.disp.size)
                continua.append(1.0)
                model_fluxes.append(_)
                model_flux_variances.append(_)
                continue

            # Get the redshift and resolution.
            z = theta.get("z", theta.get("z_{}".format(channel), 0))
            resolution = theta.get("resolution", theta.get("resolution_{}"\
                .format(channel), 0))

            if no_precomputed_binning:
                # TODO: Come back to this..
                if resolution > 0:
                    matrix = specutils.sample.resample_and_convolve(
                        self.wavelengths * (1 + z), spectrum.disp,
                        resolution)
                else:

                    matrix = specutils.sample.resample(
                        self.wavelengths * (1 + z), spectrum.disp)

                channel_fluxes = model_intensities * matrix
                channel_variance = model_variances * matrix


            else:

                # Get the pre-calculated convolution function.
                # (This will always be a callable)
                convolution_function = generate.binning_matrices[-1][i]

                t = time()
                channel_fluxes = convolution_function(
                    spectrum.disp, model_intensities, z, resolution)
                t_a = time()
                channel_variance = convolution_function(
                    spectrum.disp, model_variances, z, resolution)
                logger.debug("{0:.4f} {1:.4f}".format(t_a - t, time() - t_a))



            # Apply continuum if it is present.
            i, coeff = 0, []
            while theta.get("continuum_{0}_{1}".format(channel, i), None):
                coeff.append(theta["continuum_{0}_{1}".format(channel, i)])
                i += 1

            continuum = np.polyval(coeff[::-1], spectrum.disp) if coeff else 1.

            channel_fluxes *= continuum



            continua.append(continuum)
            model_fluxes.append(channel_fluxes)
            model_flux_variances.append(channel_variance)

        # TODO check channel fluxes are not zero at the edges.
        if kwargs.pop("full_output", False):
            # TODO do we need full_output AND return continuum?
            if kwargs.pop("__return_continuum", False):
                return (model_fluxes, model_flux_variances, matched_channels,
                    continua)
            return (model_fluxes, model_flux_variances, matched_channels)

        return model_fluxes

    # Functions that should be overwritten by subclasses..
    def _approximate_intensities(self, *args, **kwargs):
        raise NotImplementedError("this should be overwritten in a subclass")

    def _initalise_approximator(self, *args, **kwargs):
        raise NotImplementedError("this should be overwritten in a subclass")



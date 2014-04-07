# coding: utf-8

""" Handles the analysis for SCOPE """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

__all__ = ["initialise_priors", "log_likelihood", "solve"]

# Standard library
import logging
import os

# Third-party
import emcee
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import scipy.optimize

# Module
import models, utils, specutils

logger = logging.getLogger(__name__.split(".")[0])


def initialise_priors(model, observations):
    """ Initialise the priors (or initial conditions) for the analysis """

    walker_priors = []
    initial_normalisation_coefficients = {}

    walkers = model.configuration["solver"].get("walkers", 1)

    if model.configuration.has_key("priors"):
        while walkers > len(walker_priors):

            current_walker = []
            interpolated_flux = {}
            initial_normalisation_coefficients = {}
            for j, dimension in enumerate(model.dimensions):

                # Have we just finished doing the model dimensions?
                # If so then we can interpolate to a model flux
                if len(current_walker) == len(model.grid_points.dtype.names):
                    interpolated_flux = model.interpolate_flux(current_walker)

                    if np.all(~np.isfinite(interpolated_flux.values()[0])):
                        # None of the flux in the first beam are finite.

                        break

                # Jitter
                if dimension == "jitter" or dimension.startswith("jitter."):
                    # Uniform prior between 0 and 1
                    current_walker.append(random.uniform(0, 1))
                    continue

                # Implicit priors
                if dimension.startswith("normalise_observed."):

                    aperture = dimension.split(".")[1]
                    coefficient_index = int(dimension.split(".")[-1].lstrip("a"))

                    # If we're at this stage we should have grid point dimensions
                    if aperture not in initial_normalisation_coefficients:

                        index = model._mapped_apertures.index(aperture)
                        order = model.configuration["normalise_observed"][aperture]["order"]

                        spectrum = observations[index]
                        
                        # Get the full range of spectra that will be normalised
                        if "masks" in model.configuration and aperture in model.configuration["masks"]:
                            
                            ranges = np.array(model.configuration["masks"][aperture])
                            min_range, max_range = np.min(ranges), np.max(ranges)
                            range_indices = np.searchsorted(spectrum.disp, [min_range, max_range])

                            flux_indices = np.zeros(len(spectrum.disp), dtype=bool)
                            flux_indices[range_indices[0]:range_indices[1]] = True
                            flux_indices *= np.isfinite(spectrum.flux) * (spectrum.flux > 0)
                            
                            logger.debug("Normalising from {1:.0f} to {2:.0f} Angstroms in {0} aperture".format(
                                aperture, np.min(spectrum.disp[flux_indices]), np.max(spectrum.disp[flux_indices])))
                        else:
                            flux_indices = np.isfinite(spectrum.flux) * (spectrum.flux > 0) 

                        # Fit the spectrum with a polynomial of order X
                        resampled_interpolated_flux = np.interp(spectrum.disp[flux_indices], model.dispersion[aperture],
                            interpolated_flux[aperture])
                        coefficients = np.polyfit(spectrum.disp[flux_indices], spectrum.flux[flux_indices]/resampled_interpolated_flux, order)
                        
                        # Save the coefficients and variances
                        initial_normalisation_coefficients[aperture] = coefficients
                    
                    coefficient = initial_normalisation_coefficients[aperture][coefficient_index]
                    current_walker.append(coefficient)

                    continue

                # Explicit priors
                prior_value = model.configuration["priors"][dimension]
                try:
                    prior_value = float(prior_value)

                except:

                    # We probably need to evaluate this.
                    if prior_value.lower() == "uniform":
                        # Only works on stellar parameter values.
                        possible_points = model.grid_points[dimension].view(np.float)

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with uniform distribution between {1:.2e} and {2:.2e}".format(
                                dimension, np.min(possible_points), np.max(possible_points)))
                        current_walker.append(random.uniform(np.min(possible_points), np.max(possible_points)))

                    elif prior_value.lower().startswith("normal"):
                        mu, sigma = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with a normal distribution with $\mu$ = {1:.2e}, $\sigma$ = {2:.2e}".format(
                                dimension, mu, sigma))
                        current_walker.append(random.normal(mu, sigma))

                    elif prior_value.lower().startswith("uniform"):
                        minimum, maximum = map(float, prior_value.split("(")[1].rstrip(")").split(","))

                        if len(walker_priors) == 0: # Only print initialisation for the first walker
                            logger.info("Initialised {0} parameter with a uniform distribution between {1:.2e} and {2:.2e}".format(
                                dimension, minimum, maximum))
                        current_walker.append(random.uniform(minimum, maximum))

                    elif prior_value.lower() == "cross_correlate":
                        current_walker.append(random.normal(0, 100))

                        #aperture = dimension.split(".")[1]
                        #observed_spectrum = observations[model._mapped_apertures.index(aperture)]

                        #model_aperture_spectrum = specutils.Spectrum1D(model.dispersion[aperture], interpolated_flux[aperture])
                        #v_rad, u_v_rad, R = observed_spectrum.cross_correlate(model_aperture_spectrum)
                        #current_walker.append(random.normal(v_rad, u_v_rad))

                    else:
                        raise TypeError("prior type not valid for {dimension}".format(dimension=dimension))

                else:
                    if len(walker_priors) == 0: # Only print initialisation for the first walker
                        logger_fn = logger.info if walkers == 1 else logger.warn
                        logger_fn("Initialised {0} parameter as a single value: {1:.2e}".format(dimension, prior_value))

                    current_walker.append(prior_value)

            
            # Was the walker actually valid?
            if len(current_walker) > 0:

                # Add the walker
                if walkers == 1:
                    walker_priors = current_walker
                
                else:
                    walker_priors.append(current_walker)

        walker_priors = np.array(walker_priors)

    else:
        # No explicit priors given. Work it out!
        cross_correlation_peaks = np.zeros((walkers, len(model.apertures)))
        for i in xrange(walkers):
            
            current_walker = []
            interpolated_flux = {}
            normalisation_coefficients = {}
            for j, dimension in enumerate(model.dimensions):

                if dimension == "jitter" or dimension.startswith("jitter."):
                    current_walker.append(random.uniform(0, 1))
                    continue

                # Is it a model dimension?
                if dimension in model.grid_points.dtype.names:
                    # Sample uniformly
                    current_walker.append(random.uniform(*model.grid_boundaries[dimension]))
                    continue

                else:
                    # Since the grid point parameters come first in model.dimensions, the first
                    # time this else is run means we can interpolate flux
                    if len(interpolated_flux) == 0:
                        interpolated_flux.update(model.interpolate_flux(current_walker[:len(model.grid_points.dtype.names)]))
                        if np.any([np.all(~np.isfinite(value)) for value in interpolated_flux.values()]):
                            
                            k, keep_trying = 0, True
                            while keep_trying:
                                current_walker = [random.uniform(*model.grid_boundaries[dimension]) for dimension in model.grid_points.dtype.names]
                                interpolated_flux = model.interpolate_flux(current_walker[:len(model.grid_points.dtype.names)])

                                if k > walkers:
                                    raise ValueError("could not initialise priors with any finite flux points")

                                if not np.any([np.all(~np.isfinite(value)) for value in interpolated_flux.values()]):
                                    break


                # Is it a smoothing parameter? Estimate from resolution

                # Is it a velocity parameter? Estimate from cross-correlation with interpolated flux?
                if dimension.startswith("doppler_shift."):
                    aperture = dimension.split(".")[1]
                    observed_spectrum = observations[model._mapped_apertures.index(aperture)]
                    
                    template = specutils.Spectrum1D(model.dispersion[aperture], interpolated_flux[aperture])
                    vrad, u_vrad, R = observed_spectrum.cross_correlate(template)
                    current_walker.append(random.normal(vrad, u_vrad))

                    cross_correlation_peaks[i, int(model.apertures.index(aperture))] = R

                # Is it a normalisation parameter? Estimate from division with spectra at point
                elif dimension.startswith("normalise_observed."):
                    aperture, coefficient = dimension.split(".")[1:3]

                    if not normalisation_coefficients.has_key(aperture):
                        # Get the full range of spectra that will be normalised
                        observed_spectrum = observations[model._mapped_apertures.index(aperture)]
                        order = model.configuration["normalise_observed"][aperture]["order"]

                        if "masks" in model.configuration and aperture in model.configuration["masks"]:
                            
                            ranges = np.array(model.configuration["masks"][aperture])
                            min_range, max_range = np.min(ranges), np.max(ranges)
                            range_indices = np.searchsorted(observed_spectrum.disp, [min_range, max_range])

                            flux_indices = np.zeros(len(observed_spectrum.disp), dtype=bool)
                            flux_indices[range_indices[0]:range_indices[1]] = True
                            flux_indices *= np.isfinite(observed_spectrum.flux) * (observed_spectrum.flux > 0)
                            
                            logger.debug("Normalising from {1:.0f} to {2:.0f} Angstroms in {0} aperture".format(
                                aperture, np.min(observed_spectrum.disp[flux_indices]), np.max(observed_spectrum.disp[flux_indices])))
                        else:
                            flux_indices = np.isfinite(observed_spectrum.flux) * (observed_spectrum.flux > 0)

                        # Put the interpolated flux onto the same scale as the observed_spectrum
                        interpolated_resampled_flux = np.interp(observed_spectrum.disp[flux_indices], model.dispersion[aperture], interpolated_flux[aperture],
                            left=np.nan, right=np.nan)

                        # Fit the divided spectrum with a polynomial of order X
                        divided_flux = observed_spectrum.flux[flux_indices]/interpolated_resampled_flux
                        isfinite = np.isfinite(divided_flux)

                        coefficients = np.polyfit(observed_spectrum.disp[flux_indices][isfinite], divided_flux[isfinite], order)
                        normalisation_coefficients[aperture] = coefficients

                    current_walker.append(normalisation_coefficients[aperture][int(coefficient[1:])])

            walker_priors.append(map(float, current_walker))

        walker_priors = np.array(walker_priors)
        raise a

    return walker_priors


def optimise(observed_spectra, model, initial_samples=None):
    """ Optimise the model parameters prior to MCMC sampling """

    if initial_samples is None:
        initial_samples = model.configuration["solver"].get("initial_samples", 1000)

    # Define a function to fit the smoothing
    def fit_smoothing(kernel, obs_flux, obs_sigma, model_flux):
        chi_sq = (scipy.ndimage.gaussian_filter1d(model_flux, abs(kernel[0]))[np.isfinite(obs_flux)] - obs_flux[np.isfinite(obs_flux)])/obs_sigma**2
        return np.sum(chi_sq[np.isfinite(chi_sq)])

    # Define our minimisation function
    def minimisation_function(theta, model, observed, full_output=False):

        # Here theta contains only the dimensions of the grid
        interpolated_flux = model.interpolate_flux(theta)
        if sum(map(lambda x: sum(np.isfinite(x)), interpolated_flux.values())) == 0:
            return fail_value

        parameters = dict(zip(model.grid_points.dtype.names, theta))

        ndim, chi_sqs = 0, 0
        for aperture, observed_aperture in zip(model._mapped_apertures, observed):
            model_aperture = specutils.Spectrum1D(disp=model.dispersion[aperture], flux=interpolated_flux[aperture])

            # Get doppler shift
            if model.configuration["doppler_shift"][aperture]["perform"]:
                try:
                    v_rad, v_err, r = observed_aperture.cross_correlate(model_aperture)
                except:
                    return fail_value

                else:
                    if abs(v_rad) > 500:
                        return fail_value

                parameters["doppler_shift.{0}".format(aperture)] = v_rad

            else: 
                v_rad = 0

            # Put the interpolated flux onto the same scale as the observed_spectrum
            observed_rest_aperture = observed_aperture.doppler_shift(-v_rad)
            resampled_model_aperture = model_aperture.interpolate(observed_rest_aperture.disp)

            # Define masks if necessary
            if "masks" in model.configuration and aperture in model.configuration["masks"]:
                ranges = np.array(model.configuration["masks"][aperture])
                min_range, max_range = np.min(ranges), np.max(ranges)
                range_indices = np.searchsorted(observed_aperture.disp, [min_range, max_range])

                flux_indices = np.zeros(len(observed_aperture.disp), dtype=bool)
                flux_indices[range_indices[0]:range_indices[1]] = True
                flux_indices *= np.isfinite(observed_aperture.flux) * (observed_aperture.flux > 0)

            else:
                flux_indices = np.isfinite(observed_aperture.flux) * (observed_aperture.flux > 0)

            # Any normalisation to perform?
            if model.configuration["normalise_observed"][aperture]["perform"]:
                # Get normalisation coefficients
                order = model.configuration["normalise_observed"][aperture]["order"]

                # Fit the divided spectrum with a polynomial of order X
                divided_flux = (observed_rest_aperture.flux/resampled_model_aperture.flux)[flux_indices]
                isfinite = np.isfinite(divided_flux)

                coefficients = np.polyfit(observed_rest_aperture.disp[flux_indices][isfinite], divided_flux[isfinite], order)

                # Save the coefficients                
                for i, coefficient in enumerate(coefficients):
                    parameters["normalise_observed.{0}.a{1}".format(aperture, i)] = coefficient

                # Normalise the observed spectra
                observed_normalised_aperture_flux = (observed_rest_aperture.flux/np.polyval(coefficients, observed_rest_aperture.disp))[flux_indices]

            else:
                observed_normalised_aperture_flux = observed_rest_aperture.flux[flux_indices]

            # Any smoothing to perform?
            if model.configuration["smooth_model_flux"][aperture]["perform"]:

                # Do we have a kernel?
                if model.configuration["smooth_model_flux"][aperture]["kernel"] == "free":

                    # For the moment, at least some smoothing information is required in the model file
                    kernel = float(model.configuration["priors"]["smooth_model_flux.{0}.kernel".format(aperture)].split("(")[1].split(",")[0])

                    # Estimate kernel value
                    #kernel = abs(scipy.optimize.minimize(fit_smoothing, [0],
                    #    args=(observed_normalised_aperture_flux, observed_aperture.uncertainty[flux_indices], resampled_model_aperture.flux[flux_indices]))["x"])

                    # Scale kernel to a fwhm
                    #kernel *= (2 * (2*np.log(2))**0.5) * np.mean(np.diff(resampled_model_aperture.disp))
                    parameters["smooth_model_flux.{0}.kernel".format(aperture)] = kernel

                else:
                    kernel = model.configuration["smooth_model_flux"][aperture]["kernel"]

                convolved_model_aperture = resampled_model_aperture.gaussian_smooth(kernel)

            else:
                convolved_model_aperture = resampled_model_aperture

            # Calculate the chi-sq values
            chi_sq = (observed_normalised_aperture_flux - convolved_model_aperture.flux[flux_indices])**2/observed_aperture.uncertainty[flux_indices]**2
            ndim += sum(np.isfinite(chi_sq))
            chi_sqs += sum(chi_sq[np.isfinite(chi_sq)])

        r_chi_sq = chi_sqs/(ndim - len(model.grid_points.dtype.names) - 1)
        
        logger.debug(u"Optimisation is returning a reduced χ² = {0:.2f} for the point where {1}".format(
            r_chi_sq, ", ".join(["{0} = {1:.2f}".format(dim, value) for dim, value in zip(model.grid_points.dtype.names, theta)])))
        
        if full_output:
            return (r_chi_sq, parameters)

        return r_chi_sq

    fail_value = +9e99
    returned_values = []
    random_points = []

    logger.info("Random sampling...")
    while initial_samples > len(random_points):
        p0 = [np.random.uniform(*model.grid_boundaries[parameter]) for parameter in model.grid_points.dtype.names]
        
        try:
            result = minimisation_function(p0, model, observed_spectra)
        except:
            logger.exception("Failed to sample {0}".format(p0))

        else:
            returned_values.append(result)
            random_points.append(p0)

    best_index = np.argmin(returned_values)

    logger.info(u"Optimising from {0} with initial reduced χ² = {1:.2f}".format(
        ", ".join(["{0} = {1:.2f}".format(dim, value) for dim, value in zip(model.grid_points.dtype.names, random_points[best_index])]),
        returned_values[best_index]))
    result = scipy.optimize.minimize(minimisation_function, random_points[best_index],
        args=(model, observed_spectra))

    logger.info(u"Sampling from {0} with reduced χ² = {1:.2f}".format(
        ", ".join(["{0} = {1:.2f}".format(dim, value) for dim, value in zip(model.grid_points.dtype.names, result["x"])]),
        result["fun"]))

    return minimisation_function(result["x"], model, observed_spectra, full_output=True)
    

def log_prior(theta, model):
    
    parameters = dict(zip(model.dimensions, theta))
    for parameter, value in parameters.iteritems():
        # Check doppler shifts. Anything more than 500 km/s is considered implausible
        if parameter.startswith("doppler_shift.") and abs(value) > 500:
            return -np.inf

        # Check smoothing values. Any negative value is considered unrealistic
        if parameter.startswith("smooth_model_flux.") and (0 > value or value > 5):
            return -np.inf

        # Check for jitter
        if (parameter == "jitter" or parameter.startswith("jitter.")) and not (1 > value > 0):
            return -np.inf

        # Check if point is within the grid boundaries?
        if parameter in model.grid_boundaries:
            min_value, max_value = model.grid_boundaries[parameter]
            if value > max_value or min_value > value:
                return -np.inf

    return 0


def log_likelihood(theta, model, observations):
    """Calculates the likelihood that a given set of observations
    and parameters match the input models.

    parameters : list of `float`
        The free parameters to solve for. These are referenced in 
        `parameter_names`.

    observations : list of `Spectrum1D` objects
        The observed spectra.

    callback : function
        A callback to apply after completing the comparison.
    """

    if not np.isfinite(log_prior(theta, model)):
        logger.debug("Returning -inf log-likelihood because log-prior was -inf")
        return -np.inf

    parameters = dict(zip(model.dimensions, theta))

    # Prepare the observed spectra: radial velocity shift? normalisation?
    observed_spectra = model.observed_spectra(observations, **parameters)
    if observed_spectra is None:
        logger.debug("Returning -inf log-likelihood because modified observed spectra is invalid")
        return -np.inf

    # Prepare the model spectra: smoothing? re-sample to observed dispersion?
    model_spectra = model.model_spectra(observations=observed_spectra, **parameters)
    if model_spectra is None:
        logger.debug("Returning -inf log-likelihood because modified model spectra is invalid")
        return -np.inf

    # Any masks?
    # masks = model.masks(model_spectra)
    
    # Calculate chi^2 difference
    chi_sqs = {}
    for i, (aperture, modelled_spectrum, observed_spectrum) in enumerate(zip(model._mapped_apertures, model_spectra, observed_spectra)):

        inverse_variance = 1.0/(observed_spectrum.uncertainty**2)# + parameters["jitter.{0}".format(aperture)])
        chi_sq = ((observed_spectrum.flux - modelled_spectrum.flux)**2) * inverse_variance

        # Apply masks
        #chi_sq *= masks[aperture]

        # Useful_pixels of 1 indicates that we should use it, 0 indicates it was masked.
        useful_pixels = np.isfinite(chi_sq) 
        if sum(useful_pixels) == 0:
            logger.debug("Returning -np.inf log-likelihood because there were no useful pixels")
            return -np.inf

        chi_sqs[aperture] = np.sum(chi_sq[useful_pixels])# - np.log(inverse_variance[useful_pixels]))

        # Update the masks values:
        #> -2: Not interested in this region, and it was non-finite (not used).
        #> -1: Interested in this region, but it was non-finite (not used).
        #>  0: Not interested in this region, it was finite (not used).
        #>  1: Interested in this region, it was finite (used for parameter determination)
        #masks[aperture][~useful_pixels] -= 2

    likelihood = -0.5 * np.sum(chi_sqs.values())

    logger.debug("Returning log likelihood of {0:.2e} for parameters: {1}".format(likelihood,
        ", ".join(["{0} = {1:.2e}".format(name, value) for name, value in parameters.iteritems()])))  
   
    return likelihood




def solve(observed_spectra, model):
    """Analyse some spectra of a given star according to the configuration
    provided.

    Inputs
    ------
    observed_spectra : list of `Spectrum1D` objects
        Non-overlapping spectral beams of a single star.

    """

    # Check observed arms do not overlap
    observed_dispersions = [s.disp for s in observed_spectra]
    overlap = utils.find_spectral_overlap(observed_dispersions)
    if overlap is not None:
        raise ValueError("observed apertures cannot overlap in wavelength, but"
            " they do near {wavelength} Angstroms".format(wavelength=overlap))

    # Load our model if necessary
    if not isinstance(model, models.Model):
        model = models.Model(model)

    # Get the aperture mapping from observed spectra to model spectra
    # For example, which index in our list of spectra corresponds to
    # 'blue', or 'red' in our model 
    aperture_mapping = model.map_apertures(observed_spectra)
    
    # Make fmin_powell the default
    if model.configuration["solver"].get("method", "powell") == "powell":

        # Optimze the dimensions of the spectral grid only
        # Calculate velocity from cross-correlation
        # Get smoothing from power spectral density

        r_chi_sq, parameters = optimise(observed_spectra, model)
        
        return result
        
    elif model.configuration["solver"]["method"] == "emcee":

        # Ensure we have the number of walkers and steps specified in the configuration
        walkers, nsteps = model.configuration["solver"]["walkers"], \
            model.configuration["solver"]["burn"]

        lnprob0, rstate0 = None, None
        threads = model.configuration["solver"].get("threads", 1)

        mean_acceptance_fractions = np.zeros(nsteps)
        
        # Initialise priors and set up arguments for optimization
        if model.configuration["solver"].get("optimise", True):

            r_chi_sq, op_pars = optimise(observed_spectra, model)

            # Create a sample ball around the result point
            ball_point = [op_pars.get(dimension, 0) for dimension in model.dimensions]
            jitter_indices = []
            dimensional_std = []
            for di, dimension in enumerate(model.dimensions):

                if dimension in model.grid_points.dtype.names:
                    dimensional_std.append(0.10 * np.ptp(model.grid_boundaries[dimension]))

                elif dimension.startswith("doppler_shift."):
                    dimensional_std.append(10)

                elif dimension.startswith("smooth_model_flux."):
                    dimensional_std.append(0.10)

                elif dimension.startswith("normalise_observed."):
                    dimensional_std.append(0.10)

                else:
                    # Jitter, which will be over-written anyways
                    dimensional_std.append(1)
                    jitter_indices.append(di)


            p0 = emcee.utils.sample_ball(ball_point, dimensional_std, size=walkers)

            # Write over jitter priors
            for ji in jitter_indices:
                p0[:, ji] = np.random.uniform(0, 1, size=walkers)

        else:
            p0 = initialise_priors(model, observed_spectra)

        logger.info("Priors summary:")
        for i, dimension in enumerate(model.dimensions):
            if len(p0.shape) > 1 and p0.shape[1] > 1:
                logger.info("\tParameter {0} - mean: {1:.2e}, std: {2:.2e}, min: {3:.2e}, max: {4:.2e}".format(
                    dimension, np.mean(p0[:, i]), np.std(p0[:, i]), np.min(p0[:, i]), np.max(p0[:, i])))
            else:
                logger.info("\tParameter {0} - initial point: {1:.2e}".format(dimension, p0[i]))

        # Initialise the sampler
        sampler = emcee.EnsembleSampler(walkers, len(model.dimensions), log_likelihood,
            args=(model, observed_spectra), threads=threads)

        # Sample_data contains all the inputs, and the \chi^2 and L 
        # sampler_state = (pos, lnprob, state[, blobs])
        for i, sampler_state in enumerate(sampler.sample(
            p0, lnprob0=lnprob0, rstate0=rstate0, iterations=nsteps)):

            fraction_complete = (i + 1)/nsteps
            mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)

            # Announce progress
            logger.info("Sampler is >{0:.2f}% complete (step {1:.0f}) with a mean acceptance fraction of {2:.3f}".format(
                fraction_complete * 100, i + 1, mean_acceptance_fractions[i]))

            # Are we *probably* converged?
            if i > (model.configuration["solver"]["burn"] + model.configuration["solver"]["sample"]) \
            and model.configuration["solver"].get("sigma_acceptance_fraction_tolerance", 5e-4) > np.std(mean_acceptance_fractions[i - model.configuration["solver"]["sample"]:i]):
                logger.info("We have probably converged, so we are exiting the MCMC prematurely.")
                mean_acceptance_fractions = mean_acceptance_fractions[:i+1]
                break

            if mean_acceptance_fractions[i] == 0:
                logger.warn("Mean acceptance fraction is zero. Breaking out of MCMC!")
                break

        # Convert state to posteriors
        logger.info("The final mean acceptance fraction is {0:.3f}".format(mean_acceptance_fractions[-1]))

        sampler.reset()
        pos, lnprob, rstate = sampler_state[:3]

        logger.info("Sampling...")

        final_state = sampler.run_mcmc(pos, model.configuration["solver"]["sample"])
        
        # Get the quantiles
        posteriors = {}
        for parameter_name, (quantile_50, quantile_16, quantile_84) in zip(model.dimensions, 
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(sampler.chain.reshape(-1, len(model.dimensions)), [16, 50, 84], axis=0)))):
            posteriors[parameter_name] = (quantile_50, quantile_16, quantile_84)

        return (posteriors, sampler, mean_acceptance_fractions) 

            
    else:
        raise NotImplementedError("well well well, how did we find ourselves here, Mr Bond?")

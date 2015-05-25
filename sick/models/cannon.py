#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon Model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
from scipy import optimize as op

import generate
from model import Model

# Since the Cannon can algebraically solve for astrophysical parameters, given
# some rest-frame intensities, we can make use of a different optimisation
# approach. Thus, we need the sick optimise and inference modules.
from .. import (inference, optimise, specutils)

logger = logging.getLogger("sick")


class CannonModel(Model):

    def __init__(self, *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)

        model_grid = self._configuration["model_grid"]
        if "cannon_data" not in model_grid \
        and "cannon_label_vector_description" not in model_grid:
            raise ValueError("Cannon model needs either pre-trained data (as a"\
                " cannon_data key in model_grid configuration) or a label vect"\
                "or description (as cannon_label_vector_description in the mod"\
                "el_grid configuration)")

        if "cannon_data" in model_grid:
            # Pre-trained model. Load the relevant information.
            self._load_trained_model(model_grid["cannon_data"])

        else:
            # Create the cannon_label_vector.
            self._cannon_label_vector = self._interpret_label_vector(
                model_grid["cannon_label_vector_description"])

        return None


    def _load_trained_model(self, filename):

        logger.info("Loading model from {}".format(filename))
        with open(filename, "rb") as fp:
            trained_model = pickle.load(fp)

        self._cannon_coefficients, self._cannon_scatter, \
        self._cannon_label_vector, self._cannon_offsets, \
        self._cannon_grid_indices = trained_model
        return trained_model


    def train_global(self, label_vector_description=None, N=None, limits=None,
        pivot=True, **kwargs):
        """
        Train the model in a Cannon-like fashion using the grid points as labels
        and the intensities as normalised rest-frame fluxes.
        """

        lv = self._cannon_label_vector if label_vector_description is None else\
            self._interpret_label_vector(label_vector_description)
        lv_array, grid_indices, offsets = _build_label_vector_array(
            self.grid_points, lv, N, limits, pivot=pivot)
        return self._train(lv_array, grid_indices, offsets, lv, **kwargs)


    def train_local(self, closest_point, label_vector_description=None, N=None,
        pivot=True, **kwargs):
        """
        Train the model in a Cannon-like fashion using the grid points as labels
        and the intensities as normalsied rest-frame fluxes within some local
        regime.
        """

        lv = self._cannon_label_vector if label_vector_description is None else\
            self._interpret_label_vector(label_vector_description)

        # By default we will train to the nearest 10% of the grid.
        # If grid subset is a fraction, scale it to real numbers.
        if N is None:
            N = self._configuration.get("settings", {}).get("grid_subset",
                0.10)
            if 1 >= N > 0:
                N = int(np.round(N * self.grid_points.size))

        logger.debug("Using {} nearest points for local Cannon model".format(N))

        # Use closest N points.
        dtype = [(name, '<f8') for name in self.grid_points.dtype.names]
        grid_points \
            = self.grid_points.astype(dtype).view(float).reshape(-1, len(dtype))

        distance = np.sum(np.abs(grid_points - np.array(closest_point))/
            np.ptp(grid_points, axis=0), axis=1)
        grid_indices = np.argsort(distance)[:N]
        
        lv_array, _, offsets = _build_label_vector_array(
            self.grid_points[grid_indices], lv, pivot=pivot)
        return self._train(lv_array, grid_indices, offsets, lv, **kwargs)


    def train_and_save(self, model_filename, cannon_data_filename, 
        clobber=False, **kwargs):
        """
        Train the Cannon coefficients.
        """

        if any(map(os.path.exists,
            (model_filename, cannon_data_filename))) and not clobber:
            raise IOError("output file already exists")

        trained = self.train_global(**kwargs)
        with open(cannon_data_filename, "wb") as fp:
            pickle.dump(trained, fp, -1)

        self._configuration["model_grid"]["cannon_data"] = cannon_data_filename
        logger.info("Cannon coefficients pickled to {}".format(
            cannon_data_filename))

        self.save(model_filename, clobber)

        return True

    """
    def estimate(self, data, full_output=False, **kwargs):

        theta, chi_sq, dof, model_fluxes = self._estimate(data,
            full_output=True, **kwargs)

        # We can assume that the initial estimate is *reasonable*.
        # Thus, the continuum and redshift are probably OK. But the actual
        # astrophysical parameters are limited to the discretization of the
        # grid.

        # Instead we should determine the optimal astrophysical parameters
        # algebraically!
        matched_channels, missing_channels, ignore_parameters \
            = self._match_channels_to_data(data)

        observed_variances = []
        observed_intensities = []
        for channel, spectrum in zip(matched_channels, data):

            # For each channel:
            # 1) Put the observed spectrum at rest.
            z = theta.get("z", theta.get("z_{}".format(channel), 0))
            rest_observed_disp = spectrum.disp * (1 - z)

            # 2) Correct for continuum.
            j, coeffs = 0, []
            while theta.get("continuum_{0}_{1}".format(channel, j), None):
                coeffs.append(theta["continuum_{0}_{1}".format(channel, j)])
                j += 1

            # Remember: model.__call__ calculates continuum based on the
            # *observed* wavelength points, so here we do the same (e.g.,
            # not those that have potentially been corrected for redshift)
            if not coeffs: continuum = 1.0
            else:
                continuum = np.polyval(coeffs[::-1], spectrum.disp) 
            
            # 3) Put the observed data onto the self.wavelengths scale.
            # [TODO] do the resampling correctly.
            rebinned_observed_intensities = np.interp(self.wavelengths,
                rest_observed_disp, spectrum.flux / continuum,
                left=np.nan, right=np.nan)

            rebinned_observed_variances = np.interp(self.wavelengths,
                rest_observed_disp, spectrum.variance / continuum,
                left=np.nan, right=np.nan)

            observed_variances.append(rebinned_observed_variances)
            observed_intensities.append(rebinned_observed_intensities)

        # [TODO] This may be the wrong thing to do.
        observed_variances \
            = np.nanmean(np.vstack(observed_variances), axis=0)
        observed_intensities \
            = np.nanmean(np.vstack(observed_intensities), axis=0)


        # What model wavelength ranges will be required?
        wavelengths_required = []
        for channel, spectrum in zip(matched_channels, data):
            if channel is None: continue
            z = theta.get("z", theta.get("z_{}".format(channel), 0))
            wavelengths_required.append(
                [spectrum.disp[0] * (1 - z), spectrum.disp[-1] * (1 - z)])

        # (If necessary,..) train the Cannon model around the closest point.
        subset_bounds = self._initialise_approximator(closest_point=\
            [theta[p] for p in self.grid_points.dtype.names], 
            wavelengths_required=wavelengths_required, **kwargs)

        # Solve for the astrophysical parameters.
        try:
            labels = self._solve_labels(observed_intensities,
                observed_variances)

        except:
            logger.exception("Could not determine sub-grid labels in estimate:")

        else:
            theta.update(labels)

            # Update the chi_sq values etc if full_output is required.
            if full_output:
                logger.warn("Returning grid model fluxes instead")
        
        if full_output:
            return (theta, chi_sq, dof, model_fluxes)

        return theta
    """


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

        # For the Cannon model, the normalised fluxes are a multiplicative
        # operation, so we don't actually need to optimise the astrophysical
        # parameters, we only need to optimise the other ones. At each set of
        # non-astrophysical parameters we can determine the optimal
        # astrophysical parameters by a matrix inversion.
        parameters = [p for p in self.parameters if p not in ignore_parameters \
            and p not in self.grid_points.dtype.names]

        # What model wavelength ranges will be required?
        wavelengths_required = []
        for channel, spectrum in zip(matched_channels, data):
            if channel is None: continue
            z = initial_theta.get("z", initial_theta.get("z_{}".format(channel),
                0))
            wavelengths_required.append(
                [spectrum.disp[0] * (1 - z), spectrum.disp[-1] * (1 - z)])

        # (If necessary,..) train the Cannon model around the closest point.
        subset_bounds = self._initialise_approximator(closest_point=\
            [initial_theta[p] for p in self.grid_points.dtype.names], 
            wavelengths_required=wavelengths_required, **kwargs)
        
        # Get the optimisation keyword arguments.
        op_kwargs = self._configuration.get("optimise", {}).copy()
        op_kwargs.update(kwargs)

        # Get fixed keywords.
        fixed = op_kwargs.pop("fixed", {})
        if fixed is None: fixed = {}
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
        # Because astrophysical parameters are not being 'optimised', we may not
        # actually need the subset_bounds.get call, but there may be fringe
        # cases that I cannot immediately think of.
        op_kwargs["bounds"] = [input_bounds.get(p, subset_bounds.get(p, nbs)) \
            for p in parameters]

        # Apply data masks now so we don't have to do it on the fly.
        masked_data, pixels_affected = self._apply_data_mask(data)

        # Prepare the convolution functions.
        self._create_convolution_functions(matched_channels, data, parameters)

        logger.info("Optimising parameters: {0}".format(", ".join(parameters)))
        logger.info("Optimisation keywords: {0}".format(op_kwargs))

        # Create the objective function. The objective function needs to
        # continuum-normalise the observed spectra, put it at rest, then solve
        # for the best astrophysical parameters, then *generate* that spectrum.
        debug = kwargs.get("debug", False)
        def nlp(t, return_labels=False):
            # Apply fixed keywords.
            theta = dict(zip(parameters, t))
            theta.update(fixed)

            observed_variances = []
            observed_intensities = []
            for channel, spectrum in zip(matched_channels, data):

                # For each channel:
                # 1) Put the observed spectrum at rest.
                z = theta.get("z", theta.get("z_{}".format(channel), 0))
                rest_observed_disp = spectrum.disp * (1 - z)

                # 2) Correct for continuum.
                j, coeffs = 0, []
                while theta.get("continuum_{0}_{1}".format(channel, j), None):
                    coeffs.append(theta["continuum_{0}_{1}".format(channel, j)])
                    j += 1

                # Remember: model.__call__ calculates continuum based on the
                # *observed* wavelength points, so here we do the same (e.g.,
                # not those that have potentially been corrected for redshift)
                if not coeffs: continuum = 1.0
                else:
                    continuum = np.polyval(coeffs[::-1], spectrum.disp) 
                
                # 3) Put the observed data onto the self.wavelengths scale.
                # [TODO] do the resampling correctly.
                rebinned_observed_intensities = np.interp(self.wavelengths,
                    rest_observed_disp, spectrum.flux / continuum,
                    left=np.nan, right=np.nan)

                rebinned_observed_variances = np.interp(self.wavelengths,
                    rest_observed_disp, spectrum.variance / continuum,
                    left=np.nan, right=np.nan)

                observed_variances.append(rebinned_observed_variances)
                observed_intensities.append(rebinned_observed_intensities)

            # [TODO] This may be the wrong thing to do.
            observed_variances \
                = np.nanmean(np.vstack(observed_variances), axis=0)
            observed_intensities \
                = np.nanmean(np.vstack(observed_intensities), axis=0)

            # Solve for the astrophysical parameters.
            try:
                #labels = self._solve_labels(observed_intensities,
                #    observed_variances)
                labels = self._solve_labels(observed_intensities,
                    observed_variances)

            except:
                logger.exception("Could not determine labels:")
                raise
                if debug: raise
                return np.inf

            if return_labels:
                return labels

            # Put the optimal astrophysical parameters into our theta.
            theta.update(labels)

            # Make the log-probability call.
            # Note: The inference._ln_probability call takes a dictionary.
            return -inference._ln_probability(theta, self, data, debug,
                matched_channels=matched_channels)

        # Do the optimisation.
        p0 = np.array([initial_theta[p] for p in parameters])
        x_opt = optimise.minimise(nlp, p0, **op_kwargs)

        # Put the result into a usable form.
        result = dict(zip(parameters, x_opt))
        result.update(fixed)
        result.update(nlp(x_opt, return_labels=True))

        result = OrderedDict([(k, result[k]) \
            for k in self.parameters if k in result])

        if full_output:
            # Create model fluxes and calculate some metric.
            chi_sq, dof, model_fluxes = self._chi_sq(result, data)
            result = (result, chi_sq, dof, model_fluxes)

        # Remove any prepared convolution functions.
        self._destroy_convolution_functions()

        return result


    def _approximate_intensities(self, theta, data, debug=False, **kwargs):

        if kwargs.get("__intensities", None) is not None:
            # We have been passed some intensities to use. This typically
            # happens for the initial_theta estimate.
            model_wavelengths = self.wavelengths
            model_intensities = kwargs.pop("__intensities")
            model_variances = 0

        else:
            if not self._initialised:
                self._initialise_approximator(**kwargs)

            try:
                # Get the wavelengths and intensities.
                model_wavelengths = generate.wavelengths[-1]
                func = generate.intensities[-1]
                model_intensities = func([theta.get(p, np.nan) \
                    for p in self.grid_points.dtype.names]).flatten()
                model_variances = generate.variances[-1]

            except:
                if debug: raise
                # Return dummy fluxes with nans.
                return [np.nan * np.ones(s.disp.size) for s in data]

        return (model_wavelengths, model_intensities, model_variances)


    def _initialise_approximator(self, closest_point=None,
        wavelengths_required=None, **kwargs):

        # What wavelengths will be required? Assume all if not specified.
        if wavelengths_required is not None:
            mask = np.zeros(self.wavelengths.size, dtype=bool)
            for start, end in wavelengths_required:
                idx = np.clip(
                    self.wavelengths.searchsorted([start, end]) + [0, 1],
                    0, self.wavelengths.size)
                mask[idx[0]:idx[1]] = True
        else:
            mask = np.ones(self.wavelengths.size, dtype=bool)

        # Apply model masks.
        mask *= self._model_mask()

        # Do we have a globally-trained model, or should we do a local train?
        if "cannon_data" in self._configuration["model_grid"]:
            # Global Cannon.
            coeffs = self._cannon_coefficients.copy()
            coeffs[~mask, :] = np.nan
            
        else:
            # Local Cannon. Train around the closest point.
            if closest_point is None:
                raise WTFError("you want a local Cannon model but you haven't "\
                    "given me a point to train from.")

            # Do we have a local Cannon store, where we have already trained on
            # this point?
            local_store_filename = "_".join(map(str, closest_point)) + ".pkl"
            local_store_folder = self._configuration["model_grid"].get(
                "cannon_store_local", None)
            if local_store_folder is not None \
            and os.path.exists(os.path.join(local_store_folder, local_store_filename)):

                # Load that filename.
                coefficients, scatter, lv, offsets, grid_indices \
                    = self._load_trained_model(os.path.join(local_store_folder,
                        local_store_filename))

            else:
                # Locally train.
                coefficients, scatter, lv, offsets, grid_indices \
                    = self.train_local(closest_point, mask=mask, **kwargs)

                if local_store_folder is not None:
                    # Save it
                    local_store_path = os.path.join(local_store_folder,
                        local_store_filename)
                    logger.info("Saving locally trained Cannon model to {}"\
                        .format(local_store_path))
                    with open(local_store_path, "wb") as fp:
                        pickle.dump((coefficients, scatter, lv, offsets,
                            grid_indices), fp, -1)


        cannoniser = lambda pt: np.dot(self._cannon_coefficients,
            _build_label_vector_rows(self._cannon_label_vector,
                pt - self._cannon_offsets.copy()).T).flatten()

        generate.init()
        generate.wavelengths.append(self.wavelengths)
        generate.intensities.append(cannoniser)
        generate.variances[-1] = self._cannon_scatter**2

        self._initialised = True
        self._subset_bounds = {}
        for name in self.grid_points.dtype.names:
            points = self.grid_points[name][self._cannon_grid_indices]
            self._subset_bounds[name] = (min(points), max(points))

        return self._subset_bounds


    def _interpret_label_vector(self, human_readable_label_vector):

        if isinstance(human_readable_label_vector, (str, unicode)):
            human_readable_label_vector = human_readable_label_vector.split()

        # For this to work, we can't have any '^' or '*' characters in our
        # grid point names
        self._check_forbidden_label_characters("^*")

        # Some convenience functions for later.
        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1
        parameter_index = lambda d: \
            self.grid_points.dtype.names.index((d + "^").split("^")[0].strip())

        theta = []
        for description in human_readable_label_vector:

            # Is it just a parameter?
            try:
                index = self.grid_points.dtype.names.index(description.strip())

            except ValueError:
                if "*" in description:
                    # Split by * to evaluate cross-terms.
                    cross_terms = []
                    for cross_term in description.split("*"):
                        try:
                            index = parameter_index(cross_term)
                        except ValueError:
                            raise ValueError("couldn't interpret '{0}' in the "\
                                "label '{1}' as a parameter coefficient".format(
                                    *map(str.strip, (cross_term, description))))
                        cross_terms.append((index, order(cross_term)))
                    theta.append(cross_terms)

                elif "^" in description:
                    theta.append([(
                        parameter_index(description),
                        order(description)
                    )])

                else:
                    raise ValueError("could not interpret '{0}' as a parameter"\
                        " coefficient description".format(description))
            else:
                theta.append([(index, order(description))])

        logger.info("Training the Cannon model using the following description "
            "of the label vector: {0}".format(self._repr_label_vector(theta)))

        return theta


    def _repr_label_vector(self, label_vector):
        """
        Return a human-readable representation of the Cannon label vector.
        
        label_vector should be [[(1,2), (2,1)], [(1,3)]] etc.

        """

        string = ["1"]
        for cross_terms in label_vector:
            sub_string = []
            for index, order in cross_terms:
                _ = self.grid_points.dtype.names[index]
                if order > 1:
                    sub_string.append("{0}^{1}".format(_, order))
                else:
                    sub_string.append(_)
            string.append(" * ".join(sub_string))
        return " + ".join(string)


    def _solve_labels(self, normalised_flux, variance=0, **kwargs):

        if not isinstance(variance, (np.ndarray, )):
            variance = np.zeros_like(normalised_flux)

        # Which parameters are actually in the Cannon model?
        # (These are the ones we have to solve for.)
        indices = np.unique(np.hstack(
            [[term[0] for term in vector_terms if term[1] != 0] \
            for vector_terms in self._cannon_label_vector]))

        finite = np.isfinite(self._cannon_coefficients[:, 0] * normalised_flux \
            * variance)

        # Get an initial estimate of those parameters from a simple inversion.
        # (This is very much incorrect for non-linear terms).
        Cinv = 1.0 / (self._cannon_scatter[finite]**2 + variance[finite])
        A = np.dot(self._cannon_coefficients[finite, :].T,
            Cinv[:, None] * self._cannon_coefficients[finite, :])
        B = np.dot(self._cannon_coefficients[finite, :].T,
            Cinv * normalised_flux[finite])
        initial_vector_labels = np.linalg.solve(A, B)
        
        """
        Y = normalised_flux/(self._cannon_scatter**2 + variance)
        ATY = np.dot(self._cannon_coefficients[finite, :].T, Y[finite])
        CiA = self._cannon_coefficients[finite, :] \
            * np.tile(1./(self._cannon_scatter[finite]**2 + variance),
                (self._cannon_coefficients[finite, :].shape[1], 1)).T
        ATCiA = np.dot(self._cannon_coefficients[finite, :].T, CiA)
        ATCiAinv = np.linalg.inv(ATCiA)
        initial_vector_labels2 = np.dot(ATCiAinv, ATY)
        """

        # p0 contains all coefficients, but we need only the linear terms for
        # the initial estimate
        _ = np.array([i for i, vector_terms \
            in enumerate(self._cannon_label_vector) if len(vector_terms) == 1 \
            and vector_terms[0][1] == 1])
        if len(_) == 0:
            raise ValueError("no linear terms in Cannon model")

        p0 = initial_vector_labels[1 + _]

        # Create the function.
        def f(coefficients, *labels):
            return np.dot(coefficients, _build_label_vector_rows(
                self._cannon_label_vector, labels).T).flatten()

        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        kwds = kwargs.copy()
        kwds.setdefault("maxfev", 10000)
        labels, covariance = op.curve_fit(f, self._cannon_coefficients[finite],
            normalised_flux[finite], p0=p0, sigma=1.0/np.sqrt(Cinv),
            absolute_sigma=True, **kwds)

        # Since we might not have solved for every parameter, let's return a 
        # dictionary. Don't forget to apply the offsets to the inferred labels.
        labels = dict(zip(np.array(self.grid_points.dtype.names)[indices],
            labels + self._cannon_offsets))

        if full_output:
            return (labels, covariance)

        return labels


    def _train(self, lv_array, grid_indices, offsets, lv, mask=None, **kwargs):

        N_models, N_pixels = self.grid_points.size, self.wavelengths.size
        training_intensities = np.memmap(
            self._configuration["model_grid"]["intensities"],dtype="float32",
            mode="r", shape=(N_models, N_pixels))[grid_indices, :]

        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lv_array.shape[1]))
        u_intensities = kwargs.pop("u_intensities", 
            np.zeros(training_intensities.shape))

        increment = int(N_pixels / 100)
        progressbar = kwargs.pop("__progressbar", True)
        if progressbar:
            sys.stdout.write("\rTraining Cannon model from {} points:\n".format(
                grid_indices.size))
            sys.stdout.flush()

        if mask is None:
            mask = np.ones(N_pixels, dtype=bool)
        else:
            logger.debug("Using mask for Cannon training")

        for i in xrange(N_pixels):
            if progressbar and (i % increment) == 0:
                sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%".format(
                    done="=" * int((i + 1) / increment),
                    not_done=" " * int((N_pixels - i - 1)/ increment),
                    percent=100. * (i + 1)/N_pixels))
                sys.stdout.flush()

            if not mask[i]: continue
            coefficients[i, :], scatter[i] = _fit_pixel(
                training_intensities[:, i], u_intensities[:, i], lv_array)

        if progressbar:
                sys.stdout.write("\r\n")
                sys.stdout.flush()

        # Save the coefficients, biases, and the label vector description
        self._cannon_coefficients = coefficients
        self._cannon_scatter = scatter
        self._cannon_label_vector = lv
        self._cannon_offsets = offsets
        self._cannon_grid_indices = grid_indices
        del training_intensities

        return (coefficients, scatter, lv, offsets, grid_indices)


def _fit_coefficients(intensities, u_intensities, scatter, lv_array,
    full_output=False):

    # For a given scatter, return the best-fit coefficients.    
    variance = u_intensities**2 + scatter**2

    CiA = lv_array * np.tile(1./variance, (lv_array.shape[1], 1)).T
    ATCiAinv = np.linalg.inv(np.dot(lv_array.T, CiA))

    Y = intensities/variance
    ATY = np.dot(lv_array.T, Y)
    coefficients = np.dot(ATCiAinv, ATY)

    if full_output:
        return (coefficients, ATCiAinv)
    return coefficients


def _pixel_scatter_ln_likelihood(ln_scatter, intensities, u_intensities,
    lv_array, debug=False):
    
    scatter = np.exp(ln_scatter)

    try:
        # Calculate the coefficients for this level of scatter.
        coefficients = _fit_coefficients(intensities, u_intensities, scatter,
            lv_array)

    except np.linalg.linalg.LinAlgError:
        if debug: raise
        return -np.inf

    model = np.dot(coefficients, lv_array.T)
    variance = u_intensities**2 + scatter**2

    return -0.5 * np.sum((intensities - model)**2 / variance) \
        - 0.5 * np.sum(np.log(variance))


def _fit_pixel(intensities, u_intensities, lv_array, debug=False):

    # Get an initial guess of the scatter.
    scatter = np.var(intensities) - np.median(u_intensities)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(intensities)

    ln_scatter = np.log(scatter)

    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    nll = lambda ln_s, *a, **k: -_pixel_scatter_ln_likelihood(ln_s, *a, **k)
    op_scatter = np.exp(op.fmin_powell(nll, ln_scatter,
        args=(intensities, u_intensities, lv_array), disp=False))

    # Calculate the coefficients at the optimal scatter value.
    # Note that if we can't solve for the coefficients, we should just set them
    # as zero and send back a giant variance.
    try:
        coefficients = _fit_coefficients(intensities, u_intensities, op_scatter,
            lv_array)

    except np.linalg.linalg.LinAlgError:
        logger.exception("Failed to calculate coefficients")
        if debug: raise

        return (np.zeros(lv_array.shape[1]), 10e8)

    else:
        return (coefficients, op_scatter)


def _build_label_vector_rows(label_vector, labels):
    labels = np.atleast_2d(labels)
    columns = [np.ones(labels.shape[0])]
    for cross_terms in label_vector:
        column = 1
        for index, order in cross_terms:
            column *= labels[:, index]**order
        columns.append(column)

    return np.vstack(columns).T


def _build_label_vector_array(grid_points, label_vector, N=None, limits=None,
    pivot=True):

    logger.debug("Building Cannon label vector array")

    dtype = [(name, '<f8') for name in grid_points.dtype.names]
    labels = grid_points.astype(dtype).view(float).reshape(
        grid_points.size, -1)

    N_models, N_labels = labels.shape
    indices = np.ones(N_models, dtype=bool)
    if limits is not None:
        for parameter, (lower_limit, upper_limit) in limits.items():
            parameter_index = grid_points.dtype.names.index(parameter)
            indices *= (upper_limit >= labels[:, parameter_index]) * \
                (labels[:, parameter_index] >= lower_limit)

    if N is not None:
        _ = np.linspace(0, indices.sum() - 1, N, dtype=int)
        indices = np.where(indices)[0][_]   
    
    else:
        indices = np.where(indices)[0]

    labels = labels[indices]
    if pivot:
        offsets = labels.mean(axis=0)
        labels -= offsets
    else:
        offsets = np.zeros(len(self.grid_points.dtype.names))

    return (_build_label_vector_rows(label_vector, labels), indices, offsets)





"""
# Create a comparison figure
import matplotlib.pyplot as plt
from sick import specutils
    

def show_compare(idx=None):

    if idx is None:
        idx = np.random.choice(indices)
    fig, ax = plt.subplots()
    ax.plot(self.wavelengths, intensities[idx, :], c='k', label="Synthetic")
    faux_data = [specutils.Spectrum1D(disp=self.wavelengths, flux=np.ones(self.wavelengths.size))]
    theta = dict(zip(self.grid_points.dtype.names,
        np.array(list(self.grid_points[idx])) - offsets))
    ax.plot(self.wavelengths, self.__call__(data=faux_data, theta=theta)[0], c='r', label="Cannon")

    ax.set_title("{0} : {1}".format(" / ".join(self.grid_points.dtype.names), " / ".join(["{0:.2f}".format(theta[p] + offsets[i]) for i, p in enumerate(self.grid_points.dtype.names)])))

    ax.legend()

    return fig

def compare_all():

    # Generate spectra at all indices
    faux_data = [specutils.Spectrum1D(disp=self.wavelengths, flux=np.ones(self.wavelengths.size))]
    cannon_intensities = np.zeros((indices.size, self.wavelengths.size))
    true_labels = np.zeros((indices.size, len(self.grid_points.dtype.names)))
    inferred_labels = np.zeros((indices.size, len(self.grid_points.dtype.names)))
    for i, index in enumerate(indices):
        theta = dict(zip(self.grid_points.dtype.names,
            np.array(list(self.grid_points[index])) - offsets))
        cannon_intensities[i, :] = self.__call__(data=faux_data, theta=theta)[0]

        true_labels[i, :] = np.array(list(self.grid_points[index]))
        inferred_labels[i, :] = self._solve_labels(intensities[index, :])

    residuals = (intensities[indices, :] - cannon_intensities)

    for parameter in self.grid_points.dtype.names:

        unique_params = np.unique(self.grid_points[indices][parameter])

        x = []
        y = []
        for each in unique_params:
            i = (self.grid_points[indices][parameter] == each)
            x.append(each * np.ones(residuals.shape[1]))
            y.append(np.mean(np.abs(residuals[i, :]), axis=0))

        x = np.hstack(x)
        y = np.hstack(y)

        fig, ax = plt.subplots()
        ax.scatter(x, y, facecolor="k")


    for parameter in self.grid_points.dtype.names:

        idx = np.argsort(self.grid_points[indices][parameter])

        fig, ax = plt.subplots()
        image = ax.imshow(residuals[idx, :], aspect="auto")
        plt.colorbar(image)

        ax.set_title(parameter)

    for i, parameter in enumerate(self.grid_points.dtype.names):

        fig, ax = plt.subplots()
        ax.scatter(true_labels[:, i], inferred_labels[:,i], facecolor='k')
        limits = [
            min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            max([ax.get_xlim()[1], ax.get_ylim()[1]])]
        ax.plot(limits, limits, ":", c="#666666", zorder=-1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        ax.set_xlabel("True {}".format(parameter))
        ax.set_ylabel("Cannon {}".format(parameter))

    raise a

    return residuals
        #ax.imshow(residuals)


    # Compare the residuals



f = show_compare()

t = compare_all()

#fig, ax = plt.subplots()
#idx = np.argsort(self.grid_points[indices]["[alpha/Fe]"])

idx = np.random.choice(indices)
print(self.grid_points[idx])
solved = self._solve_labels(intensities[idx, :])

raise a

"""
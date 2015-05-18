#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon Model class """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import os
import sys

import numpy as np
from scipy import optimize as op

import generate
from model import Model

logger = logging.getLogger("sick")


class CannonModel(Model):

    def _solve_labels(self, flux, variance=0, **kwargs):

        # Which parameters are actually in the Cannon model?
        # (These are the ones we have to solve for.)
        parameter_indices = np.unique(np.hstack(
            [[term[0] for term in vector_terms if term[1] != 0] \
            for vector_terms in self._cannon_label_vector]))

        # Get an initial estimate of those parameters from a simple inversion.
        # (This is very much incorrect for non-linear terms).
        Cinv = 1.0 / (self._cannon_scatter**2 + variance)
        A = np.dot(self._cannon_coefficients.T,
            Cinv[:, None] * self._cannon_coefficients)
        B = np.dot(self._cannon_coefficients.T, Cinv * flux)
        initial_vector_labels = np.linalg.solve(A, B)

        # p0 contains all coefficients, but we need only the linear terms for
        # the initial estimate
        p0 = initial_vector_labels[1 + np.array([i for i, vector_terms \
            in enumerate(self._cannon_label_vector) if len(vector_terms) == 1 \
            and vector_terms[0][1] == 1])]

        # Create the function.
        def f(coefficients, *labels):
            return np.dot(coefficients, _build_label_vector_rows(
                self._cannon_label_vector, labels).T).flatten()

        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        labels, covariance = op.curve_fit(f, self._cannon_coefficients, flux,
            p0=p0, sigma=1.0/np.sqrt(Cinv), **kwargs)

        # Apply the offsets to the inferred labels.
        labels += self._cannon_offsets
        if full_output:
            return (labels, covariance)
        return labels


    def train(self, label_vector_description, N_train=None, limits=None,
        pivot=True, **kwargs):
        """
        Train the model in a Cannon-like fashion using the grid points as labels
        and the intensities as normalised rest-frame fluxes.
        """

        lv = self._interpret_label_vector(label_vector_description)
        logger.info("Training the Cannon model using the following description "
            "of the label vector: {0}".format(self._repr_label_vector(lv)))

        lv_array, indices, offsets = _build_label_vector_array(self.grid_points,
            lv, N_train, limits, pivot=pivot)

        N_models, N_pixels = self.grid_points.size, self.wavelengths.size
        intensities = np.memmap(
            self._configuration["model_grid"]["intensities"],
            dtype="float32", mode="r", shape=(N_models, N_pixels))

        training_intensities = intensities[indices, :]

        scatter = np.zeros(N_pixels)
        coefficients = np.zeros((N_pixels, lv_array.shape[1]))
        u_intensities = kwargs.pop("u_intensities", np.zeros(training_intensities.shape))

        increment = int(N_pixels / 100)
        progressbar = kwargs.pop("__progressbar", True)
        if progressbar:
            sys.stdout.write("\rTraining Cannon model:")
            sys.stdout.flush()

        for i in xrange(N_pixels):
            coefficients[i, :], scatter[i] = _fit_pixel(training_intensities[:, i],
                u_intensities[:, i], lv_array)

            if progressbar and (i % increment) == 0:
                sys.stdout.write("\r[{done}{not_done}] {percent:3.0f}%".format(
                    done="=" * int((i + 1) / increment),
                    not_done=" " * int((N_pixels - i - 1)/ increment),
                    percent=100. * (i + 1)/N_pixels))
                sys.stdout.flush()

        if progressbar:
                sys.stdout.write("\r\n")
                sys.stdout.flush()

        # Save the coefficients, biases, and the label vector description
        self._cannon_coefficients = coefficients
        self._cannon_scatter = scatter
        self._cannon_label_vector = lv
        self._cannon_offsets = offsets

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

        del intensities

        return (coefficients, scatter, lv, offsets)


    def train_and_save(self, label_vector_description, model_filename,
        cannon_coefficient_filename, N_train=None, limits=None, clobber=False):
        """
        Train the Cannon coefficients.
        """

        if any(map(os.path.exists,
            (model_filename, cannon_coefficient_filename))) and not clobber:
            raise IOError("output file already exists")

        trained = self.train(label_vector_description, N_train, limits)

        with open(cannon_coefficient_filename, "wb") as fp:
            pickle.dump(trained, fp, -1)

        self._configuration["model_grid"]["cannon_coefficients"] \
            = cannon_coefficient_filename
        
        logger.info("Cannon coefficients pickled to {}".format(
            cannon_coefficient_filename))

        self.save(model_filename, clobber)

        return True


    def _initialise_approximator(self, **kwargs):

        if not hasattr(self, "_cannon_coefficients"):
            with open(self._configuration["model_grid"]["cannon_coefficients"],
                "rb") as fp:
                self._cannon_coefficients, self._cannon_scatter, \
                self._cannon_label_vector, self._cannon_offsets \
                    = pickle.load(fp)

        # What wavelengths will be required? Assume all if not specified.
        wavelengths_required = kwargs.pop("wavelengths_required", None)
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

        coeffs = self._cannon_coefficients.copy()
        coeffs[~mask, :] = np.nan
        cannoniser = lambda pt: np.dot(self._cannon_coefficients,
            _build_label_vector_rows(self._cannon_label_vector, pt).T).flatten()

        generate.init()
        generate.wavelengths.append(self.wavelengths)
        generate.intensities.append(cannoniser)
        generate.variances[-1] = self._cannon_scatter


        self._initialised = True
        self._subset_bounds = {}
        for name in self.grid_points.dtype.names:
            points = self.grid_points[name]
            self._subset_bounds[name] = (min(points), max(points))

        return self._subset_bounds


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


    def _interpret_label_vector(self, human_readable_label_vector):

        if not isinstance(human_readable_label_vector, (list, tuple)):
            raise TypeError("theta description must be a list of strings that "\
                "describe the label combinations")

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

        return theta


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



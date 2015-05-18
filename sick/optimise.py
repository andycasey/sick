#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Generalised optimisation function. """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
from time import time

import numpy as np
import scipy.optimize as op

logger = logging.getLogger("sick")


def minimise(objective_function, p0, **kwargs):
    """
    A safe, general minimisation function.
    """

    p0 = np.array(p0)

    op_kwargs = kwargs.copy()

    # Which optimisation algorithm?
    available_methods = ("BFGS", "Nelder-Mead", "Powell", "CG", "TNC")
    method = op_kwargs.pop("method", available_methods[0])
    logger.debug("Optimising with {0} algorithm".format(method))

    op_kwargs.update({ "callback": None, "full_output": True, "retall": False })

    def _drop_forbidden_keywords(op_kwds, allowed):
        for keyword in set(op_kwds).difference(allowed):
            logger.debug("Dropping optimisation keyword {0}: {1}".format(
                keyword, op_kwds.pop(keyword)))

    t_init = time()
    if method == "Nelder-Mead":
        # Need to force some arguments.
        # Optional arguments:
        # xtol, ftol, maxiter, maxfun
        _drop_forbidden_keywords(op_kwargs, ("xtol", "ftol", "maxiter", 
            "maxfun", "full_output", "retall", "callback", "disp"))

        # Set defaults.
        op_kwargs.setdefault("disp", False)

        x_opt, f_opt, num_iter, num_funcalls, warnflag \
            = op.fmin(objective_function, p0, **op_kwargs)

        logger.debug("Number of iterations: {0}, number of function calls: {1}"\
            .format(num_iter, num_funcalls))

        if warnflag == 0:
            logger.debug("Optimisation converged after {0:.1f} seconds.".format(
                time() - t_init))
        else:
            logger.warn("Optimisation failed: {0}".format([
                "Maximum number of function evaluations.",
                "Maximum number of iterations."][warnflag - 1]))

    elif method == "Powell":
        # Optional arguments: xtol, ftol, maxiter, maxfun
        _drop_forbidden_keywords(op_kwargs, ("xtol", "ftol", "maxiter", 
            "maxfun", "retall", "callback", "full_output", "disp"))
        
        # Set defaults.
        op_kwargs.setdefault("disp", False)

        x_opt, f_opt, direc, num_iter, num_funcalls, warnflag \
            = op.fmin_powell(objective_function, p0, **op_kwargs)

        logger.debug("Number of iterations: {0}, number of function calls: {1}"\
            .format(num_iter, num_funcalls))

        if warnflag == 0:
            logger.debug("Optimisation converged after {0:.1f} seconds.".format(
                time() - t_init))
        else:
            logger.warn("Optimisation failed: {0}".format([
                "Maximum number of function evaluations.",
                "Maximum number of iterations."][warnflag - 1]))

    elif method == "CG":
        # Optional arguments: gtol, norm, epsilon, maxiter.
        _drop_forbidden_keywords(op_kwargs, ("gtol", "norm", "epsilon", 
            "maxiter", "retall", "callback", "full_output"))

        x_opt, f_opt, num_funcalls, num_gradcalls, warnflag \
            = op.fmin_cg(objective_function, p0, **op_kwargs)

        logger.debug("Number of function calls: {0}, gradient calls: {1}"\
            .format(num_funcalls, num_gradcalls))

        if warnflag == 0:
            logger.debug("Optimisation converged after {0:.1f} seconds.".format(
                time() - t_init))
        else:
            logger.debug("Optimisation failed: {0}".format([
                "Maximum number of iterations exceeded",
                "Gradient and/or function calls were not changing."
                ][warnflag - 1]))

    elif method == "BFGS":
        # Since we have at least some boundaries, this will actually call
        # the L-BFGS-B algorithm.
        # Optional arguments: m, factr, pgtol, epsilon, maxfun, maxiter

        _drop_forbidden_keywords(op_kwargs, ("m", "factr", "pgtol", 
            "approx_grad", "epsilon", "disp", "iprint", "maxfun", "maxiter"))

        # Default/required:
        op_kwargs.setdefault("factr", 10.0)
        op_kwargs["approx_grad"] = True
        
        # Because the parameters will vary by orders of magnitude, here we
        # scale everything to the initial value so that the epsilon keyword
        # makes some sense.

        scale = p0.copy()
        
        def scaled_objective_function(theta):
            return objective_function(theta.copy() * scale)

        x_opt, f_opt, info_dict = op.fmin_l_bfgs_b(scaled_objective_function,
            np.ones(p0.size), **op_kwargs)

        # Rescale.
        x_opt *= scale

        g_opt = info_dict["grad"]
        num_iter = info_dict["nit"]
        num_funcalls = info_dict["funcalls"]
        warnflag = info_dict["warnflag"]

        logger.debug("Number of iterations: {0}, function calls: {1}".format(
            num_iter, num_funcalls))
        if warnflag == 0:
            logger.debug("Optimisation converged after {0:.1f} seconds.".format(
                time() - t_init))
        else:
            logger.warn("Optimisation failed: {0}".format([
                "Too many function evaluations or iterations!",
                "{0}".format(info_dict.get("task", None))][warnflag - 1]))
    
    elif method == "TNC":
        # Optional arguments:   bounds, epsilon, scale, offset, messages, 
        #                       maxCGit, maxfun, eta, stepmx, accuracy, 
        #                       fmin, ftol, xtol, pgtol, rescale

        _drop_forbidden_keywords(op_kwargs, ("approx_grad", "bounds", "epsilon",
            "scale", "offset", "messages", "maxCGit", "maxfun", "eta", "stepmx",
            "accuracy", "fmin", "ftol", "xtol", "pgtol", "rescale", "disp"))

        # Required:
        op_kwargs["approx_grad"] = True

        x_opt, num_funcalls, rc = op.fmin_tnc(objective_function, p0, 
            **op_kwargs)
        
        rcstring = {
            -1: "Infeasible (lower bound > upper bound)",
            0: "Local minimum reached (|pg| ~= 0)",
            1: "Converged (|f_n-f_(n-1)| ~= 0)",
            2: "Converged (|x_n-x_(n-1)| ~= 0)",
            3: "Max. number of function evaluations reached",
            4: "Linear search failed",
            5: "All lower bounds are equal to the upper bounds",
            6: "Unable to progress",
            7: "User requested end of minimization"
        }[rc]

        logger.debug("Number of function calls: {0}, result: {1}".format(
            num_funcalls, rcstring))
        if rc in (1, 2):
            logger.debug("Optimisation converged after {0:.1f} seconds: {1}"\
                .format(time() - t_init, rcstring))
        else:
            logger.warn("Optimisation failed: {0}".format(rcstring))

    else:
        raise ValueError("optimisation algorithm {0} is not available "\
            "(available methods are {1})".format(
                method, ", ".join(available_methods)))


    return x_opt
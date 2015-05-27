#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""" sick, the spectroscopic inference crank. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import argparse
import cPickle as pickle
import logging
import os

import numpy as np
import yaml
import json

import sick

logger = logging.getLogger("sick")

def parser(input_args=None):
    """
    Command line parser for *sick*.
    """

    parser = argparse.ArgumentParser(
        description="sick, the spectroscopic inference crank", epilog="Use "
            "'sick-model COMMAND -h' for information on a specific command."
            " Documentation and examples available at "
            "https://github.com/andycasey/sick")

    # Create a parent subparser.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False,
        help="Vebose logging mode.")
    parent_parser.add_argument(
        "--clobber", dest="clobber", action="store_true", default=False, 
        help="Overwrite existing files if they already exist.")
    parent_parser.add_argument(
        "--debug", dest="debug", action="store_true", default=False,
        help="Enable debug mode. Any suppressed exception will be re-raised.")
    parent_parser.add_argument(
        "-o", "--output_dir", dest="output_dir", nargs="?", type=str,
        help="Directory for the files that will be created. If not given, this"
        " defaults to the current working directory.", default=os.getcwd())

    # Create subparsers.
    subparsers = parser.add_subparsers(title="command", dest="command",
        description="Specify the action to perform.")

    # Create parser for the aggregate command
    aggregate_parser = subparsers.add_parser(
        "aggregate", parents=[parent_parser],
        help="Aggregate many result files into a single tabular FITS file.")
    aggregate_parser.add_argument("output_filename", type=str, nargs=1,
        help="Output filename to aggregate results into.")
    aggregate_parser.add_argument("result_filenames", nargs="+",
        help="The YAML result filenames to combine.")
    aggregate_parser.set_defaults(func=aggregate)

    # Create parser for the estimate command
    estimate_parser = subparsers.add_parser(
        "estimate", parents=[parent_parser],
        help="Compute a point estimate of the model parameters given the data.")
    estimate_parser.add_argument(
        "model", type=str,
        help="The model filename in YAML-style formatting.")
    estimate_parser.add_argument(
        "spectrum_filenames", nargs="+",
        help="Filenames of (observed) spectroscopic data.")
    estimate_parser.add_argument(
        "--filename-prefix", "-p", dest="filename_prefix",
        type=str, help="The filename prefix to use for the output files.")
    estimate_parser.add_argument(
        "--no-plots", dest="plotting", action="store_false", default=True,
        help="Disable plotting.")
    estimate_parser.add_argument(
        "--plot-format", "-pf", dest="plot_format", action="store", type=str, 
        default="png", help="Format for output plots (default: %(default)s)")
    estimate_parser.set_defaults(func=estimate)

    # Create parser for the optimise command
    optimise_parser = subparsers.add_parser(
        "optimise", parents=[parent_parser],
        help="Optimise the model parameters, given the data.")
    optimise_parser.add_argument(
        "model", type=str,
        help="The model filename in YAML-style formatting.")
    optimise_parser.add_argument(
        "spectrum_filenames", nargs="+",
        help="Filenames of (observed) spectroscopic data.")
    optimise_parser.add_argument(
        "--filename-prefix", "-p", dest="filename_prefix", type=str,
        help="The filename prefix to use for the output files.")
    optimise_parser.add_argument(
        "--no-plots", dest="plotting", action="store_false", default=True,
        help="Disable plotting.")
    optimise_parser.add_argument(
        "--plot-format", "-pf", dest="plot_format", action="store", type=str,
        default="png", help="Format for output plots (default: %(default)s)")
    optimise_parser.set_defaults(func=optimise)

    # Create parser for the infer command
    infer_parser = subparsers.add_parser(
        "infer", parents=[parent_parser],
        help="Infer the model parameters, given the data.")
    infer_parser.add_argument(
        "model", type=str,
        help="The model filename in YAML-style formatting.")
    infer_parser.add_argument(
        "spectrum_filenames", nargs="+",
        help="Filenames of (observed) spectroscopic data.")
    infer_parser.add_argument(
        "--filename-prefix", "-p", dest="filename_prefix", type=str,
        help="The filename prefix to use for the output files.")
    infer_parser.add_argument(
        "--no-chains", dest="save_chain_files", action="store_false",
        default=True, help="Do not save the chains to disk.", )
    infer_parser.add_argument(
        "--no-plots", dest="plotting", action="store_false", default=True,
        help="Disable plotting.")
    infer_parser.add_argument(
        "--plot-format", "-pf", dest="plot_format", action="store", type=str,
        default="png", help="Format for output plots (default: %(default)s)")
    infer_parser.set_defaults(func=infer)

    args = parser.parse_args(input_args)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Create a default filename prefix based on the input filename arguments
    if args.command.lower() in ("estimate", "optimise", "infer") \
    and args.filename_prefix is None:
        args.filename_prefix = _default_output_prefix(args.spectrum_filenames)

        handler = logging.FileHandler("{}.log".format(
            os.path.join(args.output_dir, args.filename_prefix)))
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
        # Check plot formats.
        if args.plotting:

            import matplotlib.pyplot as plt
            fig = plt.figure()
            available = fig.canvas.get_supported_filetypes().keys()
            plt.close(fig)

            if args.plot_format.lower() not in available:
                raise ValueError("plotting format {0} is unavailable: Options "\
                    "are: {1}".format(
                        args.plot_format.lower(), ", ".join(available)))
    else:
        args.filename_prefix = ""

    return args


def _announce_theta(theta):
    """
    Announce theta values to the log.
    """

    c = 299792.458 # km/s
    is_a_redshift = lambda p: p == "z" or p[:2] == "z_"

    for parameter, value in theta.items():
        if isinstance(value, (int, float)):
            message = "\t{0}: {1:.3f}".format(parameter, value)
            if is_a_redshift(parameter):
                message += " [{0:.1f} km/s]".format(value * c)

        else:
            # (MAP, u_pos, u_neg)
            message = "\t{0}: {1:.3f} ({2:+.3f}, {3:+.3f})".format(parameter,
                value[0], value[1], value[2])
            if is_a_redshift(parameter):
                message += " [{0:.1f} ({1:+.1f}, {2:+.1f}) km/s]".format(
                    value[0] * c, value[1] * c, value[2] * c)
        logger.info(message)


def _prefix(args, f, char="-"):
    return os.path.join(args.output_dir, char.join([args.filename_prefix, f]))


def _ok_to_clobber(args, filenames, char="-"):

    if args.clobber:
        return True

    paths = [_prefix(args, _, char=char) for _ in filenames]
    exists = map(os.path.exists, paths)
    if any(exists):
        raise IOError("expected output filename(s) already exist and we have "
            "been told not to clobber them: {}".format(", ".join(
                [path for path, e in zip(paths, exists) if e])))
    return True


def _default_output_prefix(filenames):
    """
    Return a default filename prefix for output files based on the input files.

    :param filenames:
        The input filename(s):

    :type filenames:
        str or list of str

    :returns:
        The extensionless common prefix of the input filenames:

    :rtype:
        str
    """

    if isinstance(filenames, (str, )):
        filenames = [filenames]
    common_prefix, ext = os.path.splitext(os.path.commonprefix(
        map(os.path.basename, filenames)))
    common_prefix = common_prefix.rstrip("_-")
    return common_prefix if len(common_prefix) > 0 else "sick"


def _pre_solving(args, expected_output_files):

    # Check that it will be OK to clobber existing files.
    _ok_to_clobber(args, expected_output_files)

    # Load the model and data.
    data = map(sick.specutils.Spectrum1D.load, args.spectrum_filenames)
    model = sick.models.Model(args.model)

    logger.info("Model configuration:")
    map(logger.info, yaml.safe_dump(model._configuration, stream=None,
        allow_unicode=True, default_flow_style=False).split("\n"))

    logger.info("Model parameters ({}):".format(len(model.parameters)))
    logger.info(", ".join(model.parameters))

    # Define headers that we want in the results filename 
    metadata = {
        "model": model.hash, 
        "input_filenames": ", ".join(args.spectrum_filenames),
        "sick_version": sick.__version__,
        "headers": {}
    }

    # Get some headers from the first spectrum.
    for header in ("RA", "DEC", "COMMENT", "ELAPSED", "FIBRE_NUM", "LAT_OBS",
        "LONG_OBS", "MAGNITUDE","NAME", "OBJECT", "UTEND", "UTDATE", "UTSTART"):
        metadata["headers"][header] = data[0].headers.get(header, None)

    return (model, data, metadata)


def _write_output(filename, output):
    #with open(filename, "w+") as fp:
    #    yaml.safe_dump(metadata, stream=fp, allow_unicode=True,
    #        default_flow_style=False)

    with open(filename, "w+") as fp:
        fp.write(json.dumps(output, indent=2))
    logger.info("Results written to {}".format(filename))
    return True


def estimate(args, **kwargs):
    """
    Return a point estimate of the model parameters theta given the data.
    """

    expected_output_files = kwargs.pop("expected_output_files", None)
    if not expected_output_files:
        expected_output_files = ["estimate.yaml"]
        if args.plotting:
            expected_output_files.extend(
                ["projection-estimate.{}".format(args.plot_format)])
        
    model, data, metadata = _pre_solving(args, expected_output_files)

    try:
        theta, chisq, dof, model_fluxes = model.estimate(data, full_output=True,
            debug=args.debug)

    except:
        logger.exception("Failed to estimate model parameters")
        raise

    logger.info("Estimated model parameters are:")
    _announce_theta(theta)
    logger.info("With a chi-sq value of {0:.1f} (reduced {1:.1f}; DOF {2:.1f})"\
        .format(chisq, chisq/dof, dof))

    metadata["estimated"] = {
        "theta": theta,
        "chi_sq": chisq,
        "dof": dof,
        "r_chi_sq": chisq/dof
    }

    if args.plotting:
        fig = sick.plot.spectrum(data, model_flux=model_fluxes)
        filename = _prefix(args, "projection-estimate.{}".format(
            args.plot_format))
        fig.savefig(filename)
        logger.info("Created figure {}".format(filename))

    if kwargs.pop("__return_result", False):
        return (model, data, metadata, theta)

    # Write the result to file.
    _write_output(_prefix(args, "estimate.yaml"), metadata)
    return None



def optimise(args, **kwargs):
    """
    Optimise the model parameters.
    """

    expected_output_files = kwargs.pop("expected_output_files", None)
    if not expected_output_files:
        expected_output_files = ["optimised.yaml"]
        if args.plotting:
            expected_output_files.extend([
                "projection-estimate.{}".format(args.plot_format),
                "projection-optimised.{}".format(args.plot_format)
            ])
        
    # Estimate the model parameters, unless they are already specified.
    model = sick.models.Model(args.model)
    initial_theta = model._configuration.get("initial_theta", {})
    if len(set(model.parameters).difference(initial_theta)) == 0:
        model, data, metadata = _pre_solving(args, expected_output_files)

    else:
        model, data, metadata, initial_theta = estimate(args, 
            expected_output_files=expected_output_files, __return_result=True)

    try:
        theta, chisq, dof, model_fluxes = model.optimise(data, 
            initial_theta=initial_theta, full_output=True, debug=args.debug)

    except:
        logger.exception("Failed to optimise model parameters")
        raise

    metadata["optimised"] = {
        "theta": theta,
        "chi_sq": chisq,
        "dof": dof,
        "r_chi_sq": chisq/dof
    }

    logger.info("Optimised model parameters are:")
    _announce_theta(theta)
    logger.info("With a chi-sq value of {0:.1f} (reduced {1:.1f}; DOF {2:.1f})"\
        .format(chisq, chisq/dof, dof))

    if args.plotting:
        fig = sick.plot.spectrum(data, model_flux=model_fluxes)
        filename = _prefix(args, "projection-optimised.{}".format(
            args.plot_format))
        fig.savefig(filename)
        logger.info("Created figure {}".format(filename))

    if kwargs.pop("__return_result", False):
        return (model, data, metadata, theta)

    # Write the results to file.
    _write_output(_prefix(args, "optimised.yaml"), metadata)
    return None



def infer(args):
    """
    Infer the model parameters.
    """

    expected_output_files = ["inferred.yaml"]
    if args.plotting:
        expected_output_files.extend([each.format(args.plot_format) \
            for each in "chain.{}", "corner.{}", "acceptance-fractions.{}",
            "autocorrelation.{}"])

    # Optimise them first.
    model, data, metadata, optimised_theta = optimise(args,
        expected_output_files=expected_output_files, __return_result=True)

    # Get the inference parameters from the model configuration.
    kwargs = model._configuration.get("infer", {})
    [kwargs.pop(k, None) \
        for k in ("debug", "full_output", "initial_proposal", "data")]

    try:
        theta, chains, lnprobability, acceptance_fractions, sampler, info_dict \
            = model.infer(data, initial_proposal=optimised_theta, 
                full_output=True, debug=args.debug, 
                __keep_convolution_functions=True,
                __show_progress_bar=True, **kwargs)

    except:
        logger.exception("Failed to infer model parameters")
        raise

    metadata["inferred"] = {
        "theta": theta,
        "chi_sq": info_dict["chi_sq"],
        "dof": info_dict["dof"],
        "r_chi_sq": info_dict["chi_sq"]/info_dict["dof"]
    }

    logger.info("Inferred parameters are:")
    _announce_theta(theta)
    
    # Write the results to file.
    _write_output(_prefix(args, "inferred.yaml"), metadata)

    # Write the chains, etc to disk.
    if args.save_chain_files:
        filename = _prefix(args, "chains.pkl")
        with open(filename, "wb") as f:
            pickle.dump(
                (chains, lnprobability, acceptance_fractions, info_dict), f, -1)
        logger.info("Saved chains to {}".format(filename))

    # Make plots.
    if args.plotting:
        burn = info_dict["burn"]

        # Any truth values to plot?
        truths = model._configuration.get("truths", None)
        if truths:
            truths = [truths.get(p, np.nan) for p in model.parameters]

        # Labels?
        labels = model._configuration.get("labels", {})
        labels = [labels.get(p, p) for p in info_dict["parameters"]]
        
        # Acceptance fractions.
        fig = sick.plot.acceptance_fractions(acceptance_fractions,
            burn_in=burn)
        _ = _prefix(args, "acceptance-fractions.{}".format(args.plot_format))
        fig.savefig(_)
        logger.info("Saved acceptance fractions figure to {}".format(_))

        # Autocorrelation.
        fig = sick.plot.autocorrelation(chains, burn_in=burn)
        _ = _prefix(args, "auto-correlation.{}".format(args.plot_format))
        fig.savefig(_)
        logger.info("Saved auto-correlation figure to {}".format(_))

        # Chains.
        fig = sick.plot.chains(chains, labels=labels, burn_in=burn,
            truths=truths)
        _ = _prefix(args, "chains.{}".format(args.plot_format))
        fig.savefig(_)
        logger.info("Saved chains figure to {}".format(_))

        # Corner plots (astrophysical + all).
        N = len(model.grid_points.dtype.names)
        fig = sick.plot.corner(chains[:, burn:, :N].reshape(-1, N),
            labels=labels, truths=truths[:N] if truths else None)
        _ = _prefix(args, "corner.{}".format(args.plot_format))
        fig.savefig(_)
        logger.info("Saved corner plot (astrophysical parameters) to {}"\
            .format(_))

        if len(model.parameters) > N:
            fig = sick.plot.corner(chains[:, burn:, :].reshape(-1, len(theta)),
                labels=labels, truths=truths)
            _ = _prefix(args, "corner-all.{}".format(args.plot_format))
            fig.savefig(_)
            logger.info("Saved corner plot (all parameters) to {}".format(_))

        # Projections.
        # Note here we need to scale the chains back to redshift so the data
        # are generated properly.
        fig = sick.plot.projection(data, model, 
            chains=chains[:, burn:, :]/info_dict["scales"],
            parameters=theta.keys())
        _ = _prefix(args, "projection.{}".format(args.plot_format))
        fig.savefig(_)
        logger.info("Saved projection plot to {}".format(_))

    model._destroy_convolution_functions()

    return None


def aggregate(args):
    """
    Aggregate the results from multiple analyses into a single file.
    """

    _ok_to_clobber(args, [args.output_filename[0]], "")
    logger.debug("Aggregating to {}".format(args.output_filename[0]))

    from astropy.table import Table

    # What header keys should be passed to the final table?
    header_keys = ["RA", "DEC", "NAME", "OBJECT", "MAGNITUDE",
        "UTSTART", "UTEND", "UTDATE"]

    def load_result_file(filename, debug):
        with open(filename, "r") as fp:
            try:
                result = yaml.load(fp)
            
            except:
                logger.exception("Could not read results filename: {}".format(
                    filename))
                if debug: raise

            else:
                logger.debug("Successfully loaded results from {}".format(
                    filename))
        return result

    # Load the first set of results to get the parameter names.
    first_results = load_result_file(args.result_filenames[0], True)
    parameters = set(first_results["estimated"]["theta"].keys() \
        + first_results.get("optimised", {}).get("theta", {}).keys() \
        + first_results.get("inferred", {}).get("theta", {}).keys())

    # Grab headers that exist.
    header_keys = [k for k in header_keys if k in first_results["headers"]]

    def default_values(stage):
        keys = "chi_sq dof r_chi_sq".split()
        default = dict(zip(["{0}_{1}".format(stage, key) for key in keys],
            [np.nan] * len(keys)))
        default["theta"] = dict(zip(
            ["{0}_{1}".format(stage, parameter) for parameter in parameters],
            [np.nan] * len(parameters)))
        return default

    def extract_values(result, stage, parameter_prefixes=None):
        _ = {
            "{}_chi_sq".format(stage): result.get("chi_sq", np.nan),
            "{}_dof".format(stage): result.get("dof", np.nan),
            "{}_r_chi_sq".format(stage): result.get("r_chi_sq", np.nan)
        }
        _.update(dict(zip(
            ["{0}_{1}".format(stage, parameter) for parameter in parameters],
            [result["theta"].get(param, np.nan) for param in parameters])))

        if parameter_prefixes is not None:
            for prefix in parameter_prefixes:
                _.update(dict(zip(
                    ["_".join([stage, prefix, parameter]) \
                        for parameter in parameters],
                    [result.get("{0}_{1}".format(prefix, parameter), np.nan) \
                        for parameter in parameters])))
        return _

    rows = []
    columns = [] + header_keys + ["model", "sick_version", "results_filename"]
    for i, filename in enumerate(args.result_filenames):

        result = load_result_file(filename, debug=args.debug)
        logger.debug("Loaded results from {}".format(filename))

        # Header information.
        row = dict(zip(header_keys,
            [result["headers"].get(k, None) for k in header_keys]))
        row.update({
            "model": result["model"],
            "sick_version": result["sick_version"],
            "results_filename": filename,
        })
        
        # Include estimated values (which should always be present)
        estimated = extract_values(result["estimated"], "estimated")
        row.update(estimated)
        
        # Include optimised values, if they exist.
        optimised = extract_values(
            result.get("optimised", default_values("optimised")), "optimised")
        row.update(optimised)

        # Include inferred values, if they exist.
        inferred = extract_values(
            result.get("inferred", default_values("inferred")), "inferred",
            parameter_prefixes=["u_pos", "u_neg", "n_eff"])
        row.update(inferred)
        rows.append(row)

    for each in (estimated, optimised, inferred):
        columns += sorted(each.keys())

    table = Table(rows=rows, names=columns)
    table.write(args.output_filename[0], overwrite=args.clobber)
    logger.info("Results from {0} files aggregated and saved to {1}".format(
        len(args.result_filenames), args.output_filename[0]))


def main():
    """ Parse arguments and execute the correct sub-parser. """

    args = parser()
    return args.func(args)


if __name__ == "__main__":
    main()

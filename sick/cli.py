#!/usr/bin/env python

""" sick, the spectroscopic inference crank """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Necessary for all sub-parsers
import argparse
import logging

# Necessary for some sub-parsers
import cPickle as pickle
import json
import multiprocessing
import os
import requests
import sys
import tarfile
import yaml
from textwrap import wrap
from time import time

import numpy as np
import pyfits

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sick

CACHED_MODEL_GRID_URL = "https://raw.githubusercontent.com/andycasey/sick/master/.cached-models.json"

logger = logging.getLogger("sick")

def download_file(url):
    """ Download a file to the current working directory. """
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        progress, total = 0, int(r.headers.get("content-length"))
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                progress += len(chunk)
                complete = int(50 * progress / total)
                sys.stdout.write("\r[{0}{1}] {2:3.0f}%".format('=' * complete,
                    ' ' * (50-complete), 2*complete))
                sys.stdout.flush()
                f.write(chunk)
                f.flush()
    sys.stdout.flush()
    sys.stdout.write("\nDownload of {0} complete.\n".format(local_filename))
    return local_filename

def resume(args):
    """ Resume a MCMC simulation from a previous state. """

    if not os.path.exists(args.model):
        raise IOError("model filename {0} does not exist".format(args.model))

    if not os.path.exists(args.state_filename):
        raise IOError("state filename {0} does not exist".format(args.state_filename))

    if args.plotting:
        fig = plt.figure()
        available = fig.canvas.get_supported_filetypes().keys()
        plt.close(fig)

        if args.plot_format.lower() not in available:
            raise ValueError("plotting format '{0}' not available: Options are: {1}".format(
                args.plot_format.lower(), ", ".join(available)))

    all_spectra = [sick.specutils.Spectrum.load(filename) for filename in args.spectra]

    # Are there multiple spectra for each source?
    if args.multiple_channels:
        # If so, they should all have the same length (e.g. same number of objects)
        if len(set(map(len, all_spectra))) > 1:
            raise IOError("filenames contain different number of spectra")

        # OK, they have the same length. They are probably apertures of the same
        # stars. Let's join them properly
        sorted_spectra = []
        num_stars, num_apertures = len(all_spectra[0]), len(all_spectra)
        for i in range(num_stars):
            sorted_spectra.append([all_spectra[j][i] for j in range(num_apertures)])

        all_spectra = sorted_spectra
    else:
        all_spectra = [all_spectra]

     # Load the model
    model = sick.models.Model(args.model)

    # Display some information about the model
    logger.info("Model information: {0}".format(model))
    logger.info("Configuration:")

    # Serialise as YAML for readability
    for line in yaml.dump(model.configuration).split("\n"):
        logger.info("  {0}".format(line))

    # Load the state
    with open(args.state_filename, "rb") as fp:
        p0, lnprobability, rstate = pickle.load(fp)

    # Define headers that we want in the results filename 
    default_headers = ("RA", "DEC", "COMMENT", "ELAPSED", "FIBRE_NUM", "LAT_OBS", "LONG_OBS",
        "MAGNITUDE","NAME", "OBJECT", "RO_GAIN", "RO_NOISE", "UTDATE", "UTEND", "UTSTART", )
    default_metadata = {
        "model": model.hash, 
        "input_filenames": ", ".join(args.spectra),
        "sick_version": sick.__version__,
    }

    for i, spectra in enumerate(all_spectra, start=1):

        # Force spectra as a list
        if not isinstance(spectra, (list, tuple)):
            spectra = [spectra]

        logger.info("Starting on object #{0} (RA {1}, DEC {2} -- {3})".format(i, spectra[0].headers.get("RA", "None"),
            spectra[0].headers.get("DEC", "None"), spectra[0].headers.get("OBJECT", "Unknown")))

        # Create metadata and put header information in
        if args.skip > i - 1:
            logger.info("Skipping object #{0}".format(i))
            continue

        output = lambda x: os.path.join(args.output_dir, "-".join([args.filename_prefix, str(i), x]))

        metadata = {}
        header_columns = []
        for header in default_headers:
            if header not in spectra[0].headers: continue
            header_columns.append(header)
            metadata[header] = spectra[0].headers[header]

        # Set defaults for metadata
        metadata.update({"run_id": i})
        metadata.update(default_metadata)
        
        try:
            posteriors, sampler, info = sick.analysis.sample(spectra, model, p0=p0, lnprob0=lnprobability,
                rstate0=rstate, burn=args.burn, sample=args.sample)

        except:
            logger.exception("Failed to analyse #{0}:".format(i))
            if args.debug: raise

        else:
            
            # Update results with the posteriors
            logger.info("Posteriors:")
            max_parameter_len = max(map(len, posteriors.keys()))
            sorted_parameters = sick.utils.unique_preserved_list([] + model.parameters + posteriors.keys())
            for parameter in sorted_parameters:
                posterior_value, pos_uncertainty, neg_uncertainty = posteriors[parameter]
                logger.info("    {0}: {1:.2e} (+{2:.2e}, -{3:.2e})".format(parameter.rjust(max_parameter_len),
                    posterior_value, pos_uncertainty, neg_uncertainty))

                metadata.update({
                    parameter: posterior_value,
                    "u_maxabs_{0}".format(parameter): np.abs([neg_uncertainty, pos_uncertainty]).max(),
                    "u_pos_{0}".format(parameter): pos_uncertainty,
                    "u_neg_{0}".format(parameter): neg_uncertainty,
                })

            # Save information related to the data
            metadata.update(dict(
                [("mean_flux_channel_{0}".format(k), np.mean(spectrum.flux[np.isfinite(spectrum.flux)]).tolist()) \
                    for k, spectrum in enumerate(spectra)]
            ))

            # Save information related to the analysis
            chain_filename = output("chain.fits")
            metadata.update({
                "reduced_chi_sq": info["reduced_chi_sq"],
                "warnflag": info.get("warnflag", 0),
                "maximum_log_likelihood": np.max(info["lnprobability"][np.isfinite(info["lnprobability"])]),
                "chain_filename": chain_filename,
                "time_elapsed": info["time_elapsed"],
                "final_mean_acceptance_fraction": info["mean_acceptance_fractions"][-1],
                "model_configuration": model.configuration
            })

            # If the chain filename already exists, we will simply append to it
            chain_length = info["chain"].shape[0] * info["chain"].shape[1]
            if os.path.exists(chain_filename):
                image = pyfits.open(chain_filename)
                table_hdu = pyfits.new_table(image[1].data, nrows=len(image[1].data) + chain_length)
                offset = len(image[1].data)

            else:
                offset = 0

            walkers = model.configuration["settings"]["walkers"]
            sampled_chain = np.core.records.fromarrays(
                np.vstack([
                    np.arange(1 + offset, 1 + offset + chain_length),
                    np.arange(1 + offset, 1 + offset + chain_length) % walkers,
                    info["chain"].reshape(-1, len(model.parameters)).T,
                    info["lnprobability"].reshape(-1, 1).T
                ]),
                names=["Iteration", "Sample"] + model.parameters + ["ln_likelihood"],
                formats=["i4", "i4"] + ["f8"] * (1 + len(model.parameters)))

            if offset > 0:
                table_hdu.data[-chain_length:] = sampled_chain
                chain = np.vstack([table_hdu.data[parameter] for parameter in model.parameters]).T.reshape((walkers, -1, len(model.parameters)))
                burn_offset = int(offset / walkers)

            else:
                chain = info["chain"]
                table_hdu = pyfits.BinTableHDU(sampled_chain)
                burn_ofset = 0
            
            # Save the chain
            primary_hdu = pyfits.PrimaryHDU()
            hdulist = pyfits.HDUList([primary_hdu, table_hdu])
            hdulist.writeto(chain_filename, clobber=True)

            with open(output("result.json"), "wb+") as fp:
                json.dump(metadata, fp)

            # Close sampler pool
            if model.configuration["settings"].get("threads", 1) > 1:
                sampler.pool.close()
                sampler.pool.join()

            # Save sampler state
            with open(output("model.state"), "wb+") as fp:
                pickle.dump([sampler.chain[:, -1, :], sampler.lnprobability[:, -1], sampler.random_state], fp, -1)

            # Plot results
            if args.plotting:

                # Some filenames
                chain_plot_filename = output("chain.{0}".format(args.plot_format))
                acceptance_plot_filename = output("acceptance.{0}".format(args.plot_format))
                corner_plot_filename = output("corner.{0}".format(args.plot_format))
                pp_spectra_plot_filename = output("ml-spectra.{0}".format(args.plot_format))
                
                # Plot the mean acceptance fractions
                fig, ax = plt.subplots()
                ax.plot(info["mean_acceptance_fractions"], color="k", lw=2)
                ax.axvline(model.configuration["settings"]["burn"], linestyle=":", color="k")
                ax.set_xlabel("Step")
                ax.set_ylabel("$\langle{}a_f\\rangle$")
                fig.savefig(acceptance_plot_filename)

                # Plot the chains
                fig = sick.plot.chains(chain, labels=sick.utils.latexify(model.parameters),
                    burn_in=model.configuration["settings"]["burn"] + burn_offset, truth_color='r',
                    truths=[posteriors[parameter][0] for parameter in model.parameters])
                fig.savefig(chain_plot_filename)

                # Make a corner plot with just the astrophysical parameters
                indices = np.array([model.parameters.index(parameter) for parameter in model.grid_points.dtype.names])
                fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters))[:, indices],
                    labels=sick.utils.latexify(model.grid_points.dtype.names), truth_color='r',
                    quantiles=[.16, .50, .84], verbose=False,
                    truths=[posteriors[parameter][0] for parameter in model.grid_points.dtype.names])
                fig.savefig(corner_plot_filename)

                # Plot some spectra
                fig = sick.plot.projection(sampler, model, spectra)
                fig.savefig(pp_spectra_plot_filename)

                # Closing the figures isn't enough; matplotlib leaks memory
                plt.close("all")

            # Delete some things
            del sampler, chain, primary_hdu, table_hdu, hdulist

        # We can only resume the state from a single object
        break
    logger.info("Fin.")


def cache(args):
    """ Cache a model. """

    if not os.path.exists(args.model):
        raise IOError("model filename {0} does not exist".format(args.model))

    model = sick.models.Model(args.model)
    cached_configuration = model.cache(args.grid_points_filename, args.fluxes_filename,
        clobber=args.clobber)

    model.configuration = cached_configuration
    model.save(args.model, True)

    logging.info("Updated model filename {0} to include cached data.".format(args.model))
    return True
 

def download(args):
    """ Download requested files. """

    # Get the current list of model grids
    cached_model_list = requests.get(CACHED_MODEL_GRID_URL).json()
    message = \
        "{0}: {1}\n"\
        "\t{2}\n"\
        "\tADS Reference: {3}\n\n"

    def exit_without_download():
        sys.stdout.write("No model downloaded.\n")
        sys.exit(-1)

    if args.model_grid_name == "list":
        # Just print out all of the available ones. Follow pip-style.
        sys.stdout.write("Found {0} cached sick models online:\n\n".format(
            len(cached_model_list)))
        for model in cached_model_list:
            sys.stdout.write(message.format(model["short_name"],
                model["long_name"], "\n\t".join(wrap(model["description"])),
                model["ads_reference"]))

    else:
        # Look for the specific model.
        cached_model_names = [model["short_name"].lower() \
            for model in cached_model_list]

        requested_model_name = args.model_grid_name.lower()
        if requested_model_name not in cached_model_names:
            sys.stdout.write("No cached model matching name '{0}' found. Use " \
                "'sick get list' to retrieve the current list of cached models"\
                " available online.\n".format(requested_model_name))
            sys.exit(-1)

        else:
            # Confirm the selection
            model = cached_model_list[cached_model_names.index(requested_model_name)]
            sys.stdout.write("Found {0} model:\n\n".format(model["short_name"]))
            sys.stdout.write(message.format(model["short_name"],
                model["long_name"], "\n\t".join(wrap(model["description"])),
                model["ads_reference"]))
            sys.stdout.write("Download {0} model? [y/N]".format(model["short_name"]))
            confirm = raw_input().lower().strip()
            if len(confirm) > 0 and confirm[0] == "y":

                # Check that we won't overwrite anything.
                filename = model["download_link"].split("/")[-1]
                if os.path.exists(filename):
                    sys.stdout.write("Clobber existing file {0}? [y/N]".format(
                        filename))
                    confirm = raw_input().lower().strip()
                    if 1 > len(confirm) or confirm[0] != "y":
                        exit_without_download()

                # Once downloaded, it could overwrite files in a directory:
                if os.path.exists(model["short_name"]):
                    sys.stdout.write("This may overwrite files in pre-existing"\
                        " folder {0}/ -- is that OK? [y/N]".format(model["short_name"]))
                    confirm = raw_input().lower().strip()
                    if 1 > len(confirm) or confirm[0] != "y":
                        exit_without_download()

                # OK, download it.
                sys.stdout.write("Downloading {0} model...\n".format(
                    model["short_name"]))
                filename = download_file(model["download_link"])

                # Now untar it to a new directory.
                with tarfile.open(filename, "r") as tarball:
                    tarball.extractall(path=model["short_name"])
                sys.stdout.write("Extracted files to {0}/\n".format(model["short_name"]))

                # Remove the tarball
                os.remove(filename)

            else:
                exit_without_download()



def solve(args):
    """ Calculate posterior probability distributions for model parameters given the data. """

    if not os.path.exists(args.model):
        raise IOError("model filename {0} does not exist".format(args.model))

    if args.plotting:
        fig = plt.figure()
        available = fig.canvas.get_supported_filetypes().keys()
        plt.close(fig)

        if args.plot_format.lower() not in available:
            raise ValueError("plotting format '{0}' not available: Options are: {1}".format(
                args.plot_format.lower(), ", ".join(available)))

    all_spectra = [sick.specutils.Spectrum.load(filename) for filename in args.spectra]

    # Are there multiple spectra for each source?
    if args.multiple_channels:
        # If so, they should all have the same length (e.g. same number of objects)
        if len(set(map(len, all_spectra))) > 1:
            raise IOError("filenames contain different number of spectra")

        # OK, they have the same length. They are probably apertures of the same
        # stars. Let's join them properly
        sorted_spectra = []
        num_stars, num_apertures = len(all_spectra[0]), len(all_spectra)
        for i in range(num_stars):
            sorted_spectra.append([all_spectra[j][i] for j in range(num_apertures)])

        all_spectra = sorted_spectra
    else:
        all_spectra = [all_spectra]

    # Load the model
    model = sick.models.Model(args.model)

    # Display some information about the model
    logger.info("Model information: {0}".format(model))
    logger.info("Configuration:")

    # Serialise as YAML for readability
    for line in yaml.dump(model.configuration).split("\n"):
        logger.info("  {0}".format(line))

    # Define headers that we want in the results filename 
    default_headers = ("RA", "DEC", "COMMENT", "ELAPSED", "FIBRE_NUM", "LAT_OBS", "LONG_OBS",
        "MAGNITUDE","NAME", "OBJECT", "RO_GAIN", "RO_NOISE", "UTDATE", "UTEND", "UTSTART", )
    default_metadata = {
        "model": model.hash, 
        "input_filenames": ", ".join(args.spectra),
        "sick_version": sick.__version__,
    }

    # For each source, solve
    for i, spectra in enumerate(all_spectra, start=1):

        # Force spectra as a list
        if not isinstance(spectra, (list, tuple)):
            spectra = [spectra]

        logger.info("Starting on object #{0} (RA {1}, DEC {2} -- {3})".format(i, spectra[0].headers.get("RA", "None"),
            spectra[0].headers.get("DEC", "None"), spectra[0].headers.get("OBJECT", "Unknown")))

        # Create metadata and put header information in
        if args.skip > i - 1:
            logger.info("Skipping object #{0}".format(i))
            continue

        if args.number_to_solve != "all" and i > (int(args.number_to_solve) + args.skip):
            logger.info("We have analysed {0} spectra. Exiting..".format(args.number_to_solve))
            break

        # If there are many spectra to analyse, include the run ID in the output filenames.
        if len(all_spectra) > 1:
            output = lambda x: os.path.join(args.output_dir, "-".join([args.filename_prefix, str(i), x]))
        else:
            output = lambda x: os.path.join(args.output_dir, "-".join([args.filename_prefix, x]))

        # Does a solution already exist for this star? If so are we authorised to clobber it?
        if os.path.exists(output("result.json")) and not args.clobber:
            logger.info("Skipping object #{0} as a results file already exists ({1}) and we have been asked not to "
                "clobber it".format(i, output("result.json")))
            continue

        metadata = {}
        header_columns = []
        for header in default_headers:
            if header not in spectra[0].headers: continue
            header_columns.append(header)
            metadata[header] = spectra[0].headers[header]

        # Set defaults for metadata
        metadata.update({"run_id": i})
        metadata.update(default_metadata)
        
        try:
            posteriors, sampler, info = sick.solve(spectra, model)

        except:
            logger.exception("Failed to analyse #{0}:".format(i))
            if args.debug: raise

        else:
            
            # Update results with the posteriors
            logger.info("Posteriors:")
            max_parameter_len = max(map(len, model.parameters))
            for parameter in model.parameters:
                posterior_value, pos_uncertainty, neg_uncertainty = posteriors[parameter]
                logger.info("    {0}: {1:.2e} (+{2:.2e}, -{3:.2e})".format(parameter.rjust(max_parameter_len),
                    posterior_value, pos_uncertainty, neg_uncertainty))

                metadata.update({
                    parameter: posterior_value,
                    "u_maxabs_{0}".format(parameter): np.abs([neg_uncertainty, pos_uncertainty]).max(),
                    "u_pos_{0}".format(parameter): pos_uncertainty,
                    "u_neg_{0}".format(parameter): neg_uncertainty,
                })

            # Save information related to the data
            metadata.update(dict(
                [("mean_flux_channel_{0}".format(k), np.mean(spectrum.flux[np.isfinite(spectrum.flux)])) \
                    for k, spectrum in enumerate(spectra)]
            ))

            # Save information related to the analysis
            chain_filename = output("chain.fits")
            metadata.update({
                "reduced_chi_sq": info["reduced_chi_sq"],
                "warnflag": info["warnflag"],
                "maximum_log_likelihood": np.max(info["lnprobability"][np.isfinite(info["lnprobability"])]),
                "chain_filename": chain_filename,
                "time_elapsed": info["time_elapsed"],
                "final_mean_acceptance_fraction": info["mean_acceptance_fractions"][-1],
                "model_configuration": model.configuration
            })
            
            walkers = model.configuration["settings"]["walkers"]
            chain_length = info["chain"].shape[0] * info["chain"].shape[1]
            chain = np.core.records.fromarrays(
                np.vstack([
                    np.arange(1, 1 + chain_length),
                    np.arange(1, 1 + chain_length) % walkers,
                    info["chain"].reshape(-1, len(model.parameters)).T,
                    info["lnprobability"].reshape(-1, 1).T
                ]),
                names=["Iteration", "Sample"] + model.parameters + ["ln_likelihood"],
                formats=["i4", "i4"] + ["f8"] * (1 + len(model.parameters)))

            # Save the chain
            primary_hdu = pyfits.PrimaryHDU()
            table_hdu = pyfits.BinTableHDU(chain)
            hdulist = pyfits.HDUList([primary_hdu, table_hdu])
            hdulist.writeto(chain_filename, clobber=True)

            with open(output("result.json"), "wb+") as fp:
                json.dump(metadata, fp)

            # Close sampler pool
            if model.configuration["settings"].get("threads", 1) > 1:
                sampler.pool.close()
                sampler.pool.join()

            # Save sampler state
            with open(output("model.state"), "wb+") as fp:
                pickle.dump([sampler.chain[:, -1, :], sampler.lnprobability[:, -1], sampler.random_state], fp, -1)

            # Plot results
            if args.plotting:

                # Some filenames
                chain_plot_filename = output("chain.{0}".format(args.plot_format))
                acceptance_plot_filename = output("acceptance.{0}".format(args.plot_format))
                corner_plot_filename = output("corner.{0}".format(args.plot_format))
                pp_spectra_plot_filename = output("ml-spectra.{0}".format(args.plot_format))
                
                # Plot the mean acceptance fractions
                fig, ax = plt.subplots()
                ax.plot(info["mean_acceptance_fractions"], color="k", lw=2)
                ax.axvline(model.configuration["settings"]["burn"], linestyle=":", color="k")
                ax.set_xlabel("Step")
                ax.set_ylabel("$\langle{}a_f\\rangle$")
                fig.savefig(acceptance_plot_filename)

                # Plot the chains
                fig = sick.plot.chains(info["chain"], labels=sick.utils.latexify(model.parameters),
                    burn_in=model.configuration["settings"]["burn"], truth_color='r',
                    truths=[posteriors[parameter][0] for parameter in model.parameters])
                fig.savefig(chain_plot_filename)

                # Make a corner plot with just the astrophysical parameters
                indices = np.array([model.parameters.index(parameter) for parameter in model.grid_points.dtype.names])
                fig = sick.plot.corner(sampler.chain.reshape(-1, len(model.parameters))[:, indices],
                    labels=sick.utils.latexify(model.grid_points.dtype.names), truth_color='r',
                    quantiles=[.16, .50, .84], verbose=False,
                    truths=[posteriors[parameter][0] for parameter in model.grid_points.dtype.names])
                fig.savefig(corner_plot_filename)

                # Plot some spectra
                fig = sick.plot.projection(sampler, model, spectra)
                fig.savefig(pp_spectra_plot_filename)
                
                # Closing the figures isn't enough; matplotlib leaks memory
                plt.close("all")

            # Delete some things
            del sampler, chain, primary_hdu, table_hdu, hdulist

    logger.info("Fin.")


def aggregate(args):
    """ Aggregate JSON-formatted results into a single tabular FITS file. """

    if os.path.exists(args.output_filename) and not args.clobber:
        raise IOError("output filename {0} already exists and we have been asked not"
            " to clobber it".format(args.output_filename))
    
    # Let's just assume it all aggregates from JSON to a FITS filename
    results = []
    for filename in args.result_filenames:
        with open(filename, "r") as fp:
            try:
                results.append(json.load(fp))
            except:
                logger.exception("Could not read results filename {0}".format(filename))
                if args.debug: raise
            
            else:
                logging.debug("Successfully loaded results from {0}".format(filename))

    # Get header order and sort them
    columns = results[0].keys()

    sorted_columns = []
    # Logic: RA, DEC then all other uppercase fields in alphabetical order
    # Then any other fields that have associated u_* headers in alphabetical order, as
    # well as their u_* columns
    # Then all the others in alphabetical order
    if "RA" in columns:
        sorted_columns.append("RA")

    if "DEC" in columns:
        sorted_columns.append("DEC")

    uppercase_columns = []
    parameteral_columns = []
    for column in columns:
        if column.isupper() and column not in sorted_columns: uppercase_columns.append(column)
        elif "u_pos_{0}".format(column) in columns: parameteral_columns.append(column)
    
    uppercase_columns, parameteral_columns = map(sorted, [uppercase_columns, parameteral_columns])
    all_parameteral_columns = []
    variants = ("{0}", "u_pos_{0}", "u_neg_{0}", "u_maxabs_{0}")
    for column in parameteral_columns:
        all_parameteral_columns.extend([variant.format(column) for variant in variants])

    sorted_columns.extend(uppercase_columns)
    sorted_columns.extend(all_parameteral_columns)

    other_columns = sorted(set(columns).difference(sorted_columns))
    ignore_columns = ("model_configuration", )
    sorted_columns.extend(list(set(other_columns).difference(ignore_columns)))

    # Create data types
    formats = [("f8", "|S256")[isinstance(results[-1][each], (str, unicode))] for each in sorted_columns]

    # Create table
    results_table = np.core.records.fromrecords(
        [[result.get(each, ["|S256", np.nan][formats[i] == "f8"]) \
            for i, each in enumerate(sorted_columns)] for result in results],
        names=sorted_columns, formats=formats)

    # Write results to filename 
    primary_hdu = pyfits.PrimaryHDU()
    table_hdu = pyfits.BinTableHDU(results_table)
    hdulist = pyfits.HDUList([primary_hdu, table_hdu])
    hdulist.writeto(args.output_filename, clobber=args.clobber)

    logger.info("Successfully written results from {0} sources with {1} fields to {2}".format(
        len(results), len(results[0]), args.output_filename))


def main():
    """ Parse arguments and execute a particular subparser. """

    parser = argparse.ArgumentParser(description="sick, the spectroscopic inference crank",
        epilog="See 'sick COMMAND -h' for more information on a specific command. Documentation"
        " and examples available online at https://github.com/andycasey/sick/wiki")
    
    # Create subparsers
    subparsers = parser.add_subparsers(title="command", dest="command",
        description="Specify the action to perform.")

    # Create a parent subparser
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False,
        help="Vebose mode. Logger will print debugging messages.")
    parent_parser.add_argument("--clobber", dest="clobber", action="store_true", default=False,
        help="Overwrite existing files if they already exist.")
    parent_parser.add_argument("--debug", dest="debug", action="store_true", default=False,
        help="Debug mode. Any suppressed exception during runtime will be re-raised.")

    # Create parser for the aggregate command
    aggregate_parser = subparsers.add_parser("aggregate", parents=[parent_parser],
        help="Aggregate JSON results into a tabular format.")
    aggregate_parser.add_argument("output_filename", type=str,
        help="Output filename to aggregate results into.")
    aggregate_parser.add_argument("result_filenames", nargs="+",
        help="The JSON result filenames to combine.")
    aggregate_parser.set_defaults(func=aggregate)

    # Create parser for the download command
    download_parser = subparsers.add_parser("download", parents=[parent_parser],
        help="Download a pre-cached model from an online repository.")
    download_parser.add_argument("model_grid_name", nargs="?",
        help="The name of the pre-cached model grid to download, or 'list' (de"\
            "fault) to see what pre-cached models are available.", default="list")
    download_parser.set_defaults(func=download)

    # Create parser for the solve command
    solve_parser = subparsers.add_parser("solve", parents=[parent_parser],
        help="Compute posterior probability distributions for the model parameters, given the data.")
    solve_parser.add_argument("model", type=str,
        help="The model filename in YAML- or JSON-style formatting.")
    solve_parser.add_argument("spectra", nargs="+",
        help="Filenames of (observed) spectroscopic data.")
    solve_parser.add_argument("-o", "--output-dir", dest="output_dir", nargs="?", type=str,
        default=os.getcwd(),
        help="Directory where to save output files to.")
    solve_parser.add_argument("--filename-prefix", "-p", dest="filename_prefix", default="sick",
        help="The filename prefix to use for all output files.")
    solve_parser.add_argument("--multi-channel", "-mc", dest="multiple_channels",
        action="store_true", default=False,
        help="Use if each source has multiple spectral channels. Default is false, implying that "
        "any additional spectra refers to a different source.")
    solve_parser.add_argument("-n", "--number-to-solve", dest="number_to_solve", default="all",
        help="Specify the number of sources to solve. Default is to solve for %(default)s sources.")
    solve_parser.add_argument("-s", "--skip", dest="skip", action="store", type=int, default=0,
        help="Number of sources to skip (default: %(default)s)")
    solve_parser.add_argument("--no-plots", dest="plotting", action="store_false", default=True,
        help="Disable plotting.")
    solve_parser.add_argument("--plot-format", "-pf", dest="plot_format", action="store", type=str,
        default="jpg",
        help="Format for output plots (default: %(default)s). Available formats are (case insensitive):"
        " PDF, JPG, PNG, EPS")
    solve_parser.set_defaults(func=solve)

    # Create parser for the resume command
    resume_parser = subparsers.add_parser("resume", parents=[parent_parser],
        help="Resume MCMC simulation from a previously calculated state.")
    resume_parser.add_argument("model", type=str,
        help="The model filename in YAML- or JSON-style formatting.")
    resume_parser.add_argument("state_filename", type=str,
        help="The filename containing the pickled MCMC state.")
    resume_parser.add_argument("burn", type=int,
        help="The number of MCMC steps to burn.")
    resume_parser.add_argument("sample", type=int,
        help="The number of MCMC steps to sample after burn-in.")
    resume_parser.add_argument("spectra", nargs="+",
        help="Filenames of (observed) spectroscopic data.")
    resume_parser.add_argument("-o", "--output-dir", dest="output_dir", nargs="?", type=str,
        default=os.getcwd(),
        help="Directory where to save output files to.")
    resume_parser.add_argument("--filename-prefix", "-p", dest="filename_prefix", default="sick",
        help="The filename prefix to use for the output files.")
    resume_parser.add_argument("--multi-channel", "-mc", dest="multiple_channels",
        action="store_true", default=False,
        help="Use if each source has multiple spectral channels. Default is false, implying that "
        "any additional spectra refers to a different source.")
    resume_parser.add_argument("-s", "--skip", dest="skip", action="store", type=int, default=0,
        help="Number of sources to skip (default: %(default)s)")
    resume_parser.add_argument("--no-plots", dest="plotting", action="store_false", default=True,
        help="Disable plotting.")
    resume_parser.add_argument("--plot-format", "-pf", dest="plot_format", action="store", type=str,
        default="jpg",
        help="Format for output plots (default: %(default)s). Available formats are (case insensitive):"
        " PDF, JPG, PNG, EPS")
    resume_parser.set_defaults(func=resume)

    cache_parser = subparsers.add_parser("cache", parents=[parent_parser],
        help="Cache the provided model for fast access at run-time.")
    cache_parser.add_argument("model", type=str,
        help="The (YAML- or JSON-formatted) model filename.")
    cache_parser.add_argument("grid_points_filename", type=str,
        help="The filename to cache the grid point information to.")
    cache_parser.add_argument("fluxes_filename", type=str,
        help="The filename to cache the fluxes into.")
    cache_parser.set_defaults(func=cache)

    # Parse arguments and specify logging level
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    return args.func(args)

if __name__ == "__main__":
    main()

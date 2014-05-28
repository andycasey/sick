#!/usr/bin/env python

""" sick, the spectroscopic inference crank """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

# Necessary for all sub-parsers
import argparse
import logging

# Necessary for some sub-parsers
import acor
import cPickle as pickle
import json
import multiprocessing
import os
import yaml
from time import time

import numpy as np
import pyfits

import sick

# Initialise logging
logger = logging.getLogger("sick")

def download(args):
    """
    Download requested files.
    """

    raise NotImplementedError


def solve(args):
    """
    Calculate posterior probability distributions for model parameters given the data.
    """

    if not os.path.exists(args.model):
        raise IOError("model filename {0} does not exist".format(args.model))

    available = ("pdf", "jpg", "png", "eps")
    if args.plotting and args.plot_format.lower() not in available:
        raise ValueError("plotting format '{0}' not available. Options: {1}".format(
            args.plot_format.lower(), ", ".join(available)))

    all_spectra = [sick.Spectrum.load(filename) for filename in args.spectra]

    if args.plotting:
        # Import plotting dependencies 
        import matplotlib as mpl
        mpl.use("Agg")

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        import triangle

    # Are there multiple spectra for each source?
    if args.multiple_channels:
        # If so, they should all have the same length (e.g. same number of objects)
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

    # Load the model
    model = sick.models.Model(args.model)

    # Display some information about the model
    logger.info("Model information: {0}".format(model))
    logger.info("Configuration:")

    # Serialise as YAML for readability
    for line in yaml.dump(model.configuration).split("\n"):
        logger.info("\t{}".format(line))

    # Define headers that we want in the results filename 
    default_headers = ("RA", "DEC", "COMMENT", "ELAPSED", "FIBRE_NUM", "LAT_OBS", "LONG_OBS",
        "MAGNITUDE","NAME", "OBJECT", "RO_GAIN", "RO_NOISE", "UTDATE", "UTEND", "UTSTART", )
    default_metadata = {
        "model": model.hash, 
        "input_filenames": ", ".join(args.spectra),
        "sick_version": sick.__version__,
        "walkers": model.configuration["solver"]["walkers"],
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

        # Does a solution already exist for this star? If so are we authorised to clobber it?
        output = lambda x: os.path.join(args.output_dir, "-".join([args.filename_prefix, str(i), x]))
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
        
        t_init = time()
        try:
            posteriors, sampler, info = sick.solve(spectra, model)

        except:
            logger.exception("Failed to analyse #{0}:".format(i))
            if args.debug: raise

            try: all_results
            except NameError:
                # Cannot add to results because we don't know all the model dimensions. We must continue
                # until we find one result that works
                continue
            else:
                row = np.core.records.fromrecords([[metadata.get(key, np.nan) for key in metadata_columns]],
                    dtype=all_results.dtype)
                all_results = np.append(all_results, row)

        else:
            t_elapsed = int(time() - t_init)
            
            # Update results with the posteriors
            logger.info("Posteriors:")
            for dimension, (posterior_value, pos_uncertainty, neg_uncertainty) in posteriors.iteritems():
                logger.info("\t{0:15s}: {1:.2f} (+{2:.2f}, -{3:.2f})".format(dimension, posterior_value,
                    pos_uncertainty, neg_uncertainty))

                metadata.update({
                    dimension: posterior_value,
                    "u_maxabs_{0}".format(dimension): np.abs([neg_uncertainty, pos_uncertainty]).max(),
                    "u_pos_{0}".format(dimension): pos_uncertainty,
                    "u_neg_{0}".format(dimension): neg_uncertainty,
                })

            # Set some filename variables
            chain_filename = output("chain.fits")
            pp_observed_spectra_filenames = [output("pp-obs-{}.fits".format(channel)) \
                for channel in model.channels]
            pp_modelled_spectra_filenames = [output("pp-mod-{}.fits".format(channel)) \
                for channel in model.channels]
            
            # Save information related to the analysis
            metadata.update({
                "warnflag": info["warnflag"],
                "maximum_log_likelihood": np.max(info["lnprobability"][np.isfinite(info["lnprobability"])]),
                "chain_filename": chain_filename,
                "pp_observed_spectra_filenames": ",".join(pp_observed_spectra_filenames),
                "pp_modelled_spectra_filenames": ",".join(pp_modelled_spectra_filenames),
                "time_elapsed": t_elapsed,
                "final_mean_acceptance_fraction": info["mean_acceptance_fractions"][-1],
            })
            
            # Append an sample and step number
            walkers = model.configuration["solver"]["walkers"]
            chain_length = info["chain"].shape[0] * info["chain"].shape[1]
            chain = np.core.records.fromarrays(
                np.vstack([
                    np.arange(1, 1 + chain_length),
                    np.arange(1, 1 + chain_length) % walkers,
                    info["chain"].reshape(-1, len(model.dimensions)).T,
                    info["lnprobability"].reshape(-1, 1).T
                ]),
                names=["Iteration", "Sample"] + model.dimensions + ["ln_likelihood"],
                formats=["i4", "i4"] + ["f8"] * (1 + len(model.dimensions)))

            # Save the chain
            primary_hdu = pyfits.PrimaryHDU()
            table_hdu = pyfits.BinTableHDU(chain)
            hdulist = pyfits.HDUList([primary_hdu, table_hdu])
            hdulist.writeto(chain_filename, clobber=True)

            with open(output("result.json"), "wb") as fp:
                json.dump(metadata, fp)

            # Close sampler pool
            if model.configuration["solver"].get("threads", 1) > 1:
                sampler.pool.close()
                sampler.pool.join()

            # Save sampler state
            with open(output("model.state"), "wb") as fp:
                pickle.dump([sampler.chain[:, -1, :], sampler.lnprobability[:, -1], sampler.random_state], fp, -1)

            # Get the most likely sample
            ml_index = np.argmax(sampler.lnprobability.reshape(-1))
            ml_parameters = dict(zip(model.dimensions, sampler.chain.reshape(-1, len(model.dimensions))[ml_index]))
            ml_metadata = metadata.copy()
            ml_metadata.update(ml_parameters)
            with open(output("pp.json"), "wb") as fp:
                json.dump(ml_metadata, fp)

            pp_model_fluxes = model(observations=spectra, **dict(zip(model.dimensions, [posteriors[each][0] for each in model.dimensions])))
            ml_model_fluxes = model(observations=spectra, **ml_parameters)
            #[spectrum.save(filename) for spectrum, filename in zip(spectra, pp_observed_spectra_filenames)]
            #[spectrum.save(filename) for spectrum, filename in zip(pp_modelled_spectra, pp_modelled_spectra_filenames)]

            # Plot results
            if args.plotting:

                colours = """#002F2F #046380 #A7A373 #8E2800 #B64926 #FFB03B #468966 #A989CD #1695A3 #EB7F00 #2E0927
                    #D90000 #04756F #FF8C00 #2185C5 #3E454C""".split()

                # Some filenames
                autocorrelation_plot_filename = output("acor.{}".format(args.plot_format))
                acceptance_plot_filename = output("acceptance.{}".format(args.plot_format))
                corner_plot_filename = output("corner.{}".format(args.plot_format))
                pp_spectra_plot_filename = output("ml-spectra.{}".format(args.plot_format))

                # Plot the chains
                ndim = len(model.dimensions)
                chain_to_plot = sampler.chain.reshape(-1, ndim)
                chains_per_plot = len(model.grid_points.dtype.names)

                n = 1
                for j in xrange(ndim):

                    if j % chains_per_plot == 0:
                        if j > 0:
                            [axes.xaxis.set_ticklabels([]) for axes in fig.axes[:-1]]
                            ax.set_xlabel("Iteration")
                            fig.savefig(output("chain-{0}.{1}".format(n, args.plot_format)))
                            n += 1
                        fig = plt.figure()

                    ax = fig.add_subplot(chains_per_plot, 1, (1 + j) % chains_per_plot)
                    for k in xrange(walkers):
                        ax.plot(range(1, 1 + len(info["mean_acceptance_fractions"])), info["chain"][k, :, j],
                            c=colours[j % len(colours)], alpha=0.5)
                    ax.axvline(model.configuration["solver"]["burn"], ymin=0, ymax=1, linestyle=":", c="k")
                    ax.set_ylabel(sick.utils.latexify([model.dimensions[j]])[0])
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                    ax.set_xlim(0, ax.get_xlim()[1])

                [axes.xaxis.set_ticklabels([]) for axes in fig.axes[:-1]]
                ax.set_xlabel("Step Number")
                fig.savefig(output("chain-{0}.{1}".format(n, args.plot_format)))

                # Plot the mean acceptance fractions
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(range(1, 1 + len(info["mean_acceptance_fractions"])), info["mean_acceptance_fractions"], "k", lw=2)
                ax.axvline(model.configuration["solver"]["burn"], ymin=0, ymax=1, linestyle=":", c="k")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("$\langle{}f_a\\rangle$")
                fig.savefig(acceptance_plot_filename)

                # Plot the autocorrelation function
                fig = plt.figure()
                ax = fig.add_subplot(111)
                rho_tau = [acor.function(np.mean(info["chain"][:, :, j], axis=0)) \
                    for j in xrange(len(model.dimensions))]

                for i, dimension in enumerate(model.dimensions):
                    ax.plot(range(1, 1 + len(info["mean_acceptance_fractions"])), rho_tau[i], c=colours[i % len(colours)], lw=2)

                ax.axvline(model.configuration["solver"]["burn"], ymin=0, ymax=1, linestyle=":", c="k")
                ax.axhline(0, c="#666666")
                ax.set_xlabel("$\\tau$")
                ax.set_ylabel("$\\rho(\\tau)$")
                fig.savefig(autocorrelation_plot_filename)

                # Make a corner plot with just the parameters of interest
                indices = np.array([model.dimensions.index(dimension) for dimension in model.grid_points.dtype.names])
                fig = triangle.corner(sampler.chain.reshape(-1, len(model.dimensions))[:, indices],
                    labels=sick.utils.latexify(model.grid_points.dtype.names),
                    quantiles=[.16, .50, .84], verbose=False)
                fig.savefig(corner_plot_filename)

                # Plot spectra
                # Sample spectra from posterior
                n_sample_posterior = 100
                sample_posterior_fluxes = []
                reshaped_chain = sampler.chain.reshape(-1, len(model.dimensions))
                for i in xrange(n_sample_posterior):

                    theta = reshaped_chain[np.random.randint(0, len(reshaped_chain))]
                    try:
                        sample_posterior_fluxes.append(model(observations=spectra, 
                            **dict(zip(model.dimensions, theta))))
                    except:
                        continue

                fig = plt.figure()
                for i, (channel, pp_model_flux, ml_model_flux, observed_aperture) \
                in enumerate(zip(model.channels, pp_model_fluxes, ml_model_fluxes, spectra)):

                    ax = fig.add_subplot(len(spectra), 1, i+1)

                    # Get the full boundaries
                    fmd = np.isfinite(pp_model_flux)
                    fod = np.isfinite(observed_aperture.flux)
                    full_extent = [
                        np.max([observed_aperture.disp[fmd][0], observed_aperture.disp[fod][0]]),
                        np.min([observed_aperture.disp[fmd][-1], observed_aperture.disp[fod][-1]])
                    ]

                    ax.plot(observed_aperture.disp, pp_model_flux, 'b', zorder=1, label="PP")
                    ax.plot(observed_aperture.disp, ml_model_flux, 'r', zorder=1, label="ML")
                    ax.plot(observed_aperture.disp, observed_aperture.flux, 'k', zorder=100)

                    for sample in sample_posterior_fluxes:
                        ax.plot(observed_aperture.disp, sample[i], c="#bbbbbb", zorder=-1)
                    
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                    ax.set_ylabel("Flux, $F_\lambda$")

                    ax.set_xlim(full_extent)

                    indices = observed_aperture.disp.searchsorted(full_extent)
                    relevant_fluxes = observed_aperture.flux[indices[0]:indices[1]]

                    ax.set_ylim(
                        0.90 * relevant_fluxes[np.isfinite(relevant_fluxes)].min(),
                        1.10 * relevant_fluxes[np.isfinite(relevant_fluxes)].max()
                    )
                    if i == 0:
                        ax.legend(frameon=False)

                ax.set_xlabel("Wavelength, $\lambda$")
                fig.savefig(pp_spectra_plot_filename)

                # Closing the figures isn't enough; matplotlib leaks memory
                plt.close("all")

            # Delete some things
            del sampler, chain, primary_hdu, table_hdu, hdulist

    logger.info("Fin.")


def aggregate(args):
    """
    Aggregate JSON-formatted results into a single tabular file.
    """

    if os.path.exists(args.output_filename) and not args.clobber:
        raise IOError("output filename {0} already exists and we have been asked not to clobber it".format(
            args.output_filename))
    
    # Let's just assume it all aggregates from JSON to a FITS filename
    results = []
    for filename in args.result_filenames:
        with open(filename, "r") as fp:
            try:
                results.append(json.load(fp))
            except:
                logger.exception("Could not read results filename {0}".format(filename))
                if args.debug: raise
                
    # Get header order and sort them
    columns = results[0].keys()

    sorted_columns = []
    # Logic: RA, DEC then all other uppercase fields in alphabetical order
    # Then any other fields that have associated u_* headers in alphabetical order, as well as their u_* columns
    # Then all the others in alphabetical order
    if "RA" in columns:
        sorted_columns.append("RA")

    if "DEC" in columns:
        sorted_columns.append("DEC")

    uppercase_columns = []
    dimensional_columns = []
    for column in columns:
        if column.isupper() and column not in sorted_columns: uppercase_columns.append(column)
        elif "u_pos_{0}".format(column) in columns: dimensional_columns.append(column)
    
    uppercase_columns, dimensional_columns = map(sorted, [uppercase_columns, dimensional_columns])
    all_dimensional_columns = []
    variants = ("{0}", "u_pos_{0}", "u_neg_{0}", "u_maxabs_{0}")
    for column in dimensional_columns:
        all_dimensional_columns.extend([variant.format(column) for variant in variants])

    sorted_columns.extend(uppercase_columns)
    sorted_columns.extend(all_dimensional_columns)

    other_columns = sorted(set(columns).difference(sorted_columns))
    sorted_columns.extend(list(other_columns))

    # Create data types
    formats = [("f8", "|S256")[isinstance(results[0][each], str)] for each in sorted_columns]

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

    logger.info("Successfully written {0} results with {1} fields to {2}".format(
        len(results), len(results[0]), args.output_filename))


def main():
    """ 
    Parse arguments and execute a particular subparser.
    """

    parser = argparse.ArgumentParser(description="sick, the spectroscopic inference crank",
        epilog="See 'sick COMMAND -h' for more information on a specific command. Documentation"
        " and examples available online at http://astrowizici.st/sick")
    
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
    aggregate_parser.add_argument("JSON_result_filenames", nargs="+",
        help="The JSON result filenames to combine.")
    aggregate_parser.set_defaults(func=aggregate)

    # Create parser for the get command
    get_parser = subparsers.add_parser("get", parents=[parent_parser],
        help="Retrieve specific model or data files (e.g., example files) from online repository.")
    get_parser.set_defaults(func=download)

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

    # Parse arguments and specify logging level
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    return args.func(args)

if __name__ == "__main__":
    main()
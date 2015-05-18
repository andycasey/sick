#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

""" Command line interface for *sick* model management. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import argparse
import logging
import os

logger = logging.getLogger("sick")

def parser(input_args=None):
    """
    Command line parser for dealing with *sick* models.
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

    # Sub-parser for the create model command.
    # create <MODEL_NAME> <GRID_POINTS_FILENAME> <DISPERSION_FILENAME*>
    creator = subparsers.add_parser("create", parents=[parent_parser],
        help="Create a model from existing wavelength and flux files.")
    creator.add_argument("model_name", type=str,
        help="Name for the model to be created. This will form the prefix of "
        "filenames for the model.")
    creator.add_argument("grid_points_filename", type=str,
        help="Filename containing a header with parameter and channel names, "
        "and a list of grid points and associated flux filenames.")
    creator.add_argument("wavelength_filenames", type=str, nargs="+",
        help="Filenames containing the wavelengths for the channels referred "
        "to in the header of `grid_points_filename`.")
    creator.set_defaults(func=create)

    # Sub-parser for the recast model command.
    # recast <MODEL_NAME> <ORIGINAL_MODEL_NAME> <CHANNEL_DESCRIPTION_FILENAME>
    recaster = subparsers.add_parser("recast", parents=[parent_parser],
        help="Take an existing model and create a new model with different "
            "wavelength ranges, binning, and/or resolving powers.")
    recaster.add_argument("model_name", type=str,
        help="Name for the model to be created. This will form the prefix of "
        "filenames for the model.")
    recaster.add_argument("original_model_filename", type=str,
        help="Path of the YAML-formatted model filename to use as a template.")
    recaster.add_argument("channel_description_filename", type=str,
        help="Path to a filename containing a description of the channels to "
        "be cast.")
    recaster.set_defaults(func=recast)

    # Sub-parser for the download model command.
    # download <MODEL_NAME>
    download_parser = subparsers.add_parser("download", parents=[parent_parser],
        help="Download a model from an online repository.")
    download_parser.add_argument("model_name", nargs="?",
        help="The name of the pre-cached model grid to download, or 'list' "
            "(default) to see what models are available.", default="list")
    download_parser.set_defaults(func=download)

    args = parser.parse_args(input_args)
    
    # Setup logging, bro.
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    return args

    
def main():
    """ Parse arguments and execute the correct sub-parser. """

    args = parser()
    return args.func(args)


def download(*args):
    raise NotImplementedError


def recast(args):
    """ Create a model by recasting an existing model. """

    import yaml
    from numpy import arange
    from sick.models import Model
    
    # Load in the original model.
    model = Model(args.original_model_filename)

    # Load in the channel information
    with open(args.channel_description_filename, "r") as fp:
        channel_description = yaml.load(fp)

    # Put the channel description into the format we need.
    channels = {}
    required_keys = ("wavelength_start", "wavelength_end", "wavelength_step")
    for name, descr in channel_description.iteritems():
        # Description must have:
        # wavelength_start, wavelength_end, wavelength_step,
        # optional: resolution.
        missing_keys = set(required_keys).difference(descr)
        if missing_keys:
            raise KeyError("channel {0} is missing the following key(s): {1}"\
                .format(name, ", ".join(missing_keys)))

        if descr["wavelength_start"] >= descr["wavelength_end"]:
            raise ValueError("wavelength start value for channel {0} is greater"
                " than or equal to the end value ({1} >= {2})".format(
                    name, descr["wavelength_start"], descr["wavelength_end"]))
        if 0 >= descr["wavelength_step"]:
            raise ValueError("wavelength step for channel {} must be a positive"
                " value".format(name))

        wavelengths = arange(descr["wavelength_start"], descr["wavelength_end"],
            descr["wavelength_step"])

        if 2 > wavelengths.size:
            raise ValueError("number of wavelength points requested for channel"
                "{} is less than two".format(name))

        resolution = descr.get("resolution", float("inf"))
        channels[name] = (wavelengths, resolution)

    return model.cast(args.model_name, channels, output_dir=args.output_dir,
        clobber=args.clobber, __progressbar=True)


def create(args):
    """ Create a model from wavelength and flux files. """

    from sick.models.create import create 
    return create(os.path.join(args.output_dir, args.model_name),
        args.grid_points_filename, args.wavelength_filenames,
        clobber=args.clobber)


if __name__ == "__main__":
    main()

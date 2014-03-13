#!/usr/bin/python

""" Script to run SCOPE from the command line """

from __future__ import division, print_function

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import os
import argparse


parser = argparse.ArgumentParser(description="infer stellar parameters from spectra")
parser.add_argument("model", type=str, help="YAML- or JSON-style model filename")

parser.add_argument("spectra", nargs="+")

parser.add_argument("-o", "--output-dir", dest="output", nargs="?", help="directory for output files",
	default=os.getcwd())
parser.add_argument("--no-plots", dest="plotting", action="store_false", default=True)
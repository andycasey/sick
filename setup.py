# coding: utf-8

""" sick, the spectroscopic inference crank """ 

import os
import re
import sys

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

major, minor1, minor2, release, serial =  sys.version_info
open_kwargs = {"encoding": "utf-8"} if major >= 3 else {}

def readfile(filename):
    with open(filename, **open_kwargs) as fp:
        contents = fp.read()
    return contents

version_regex = re.compile("__version__ = \"(.*?)\"")
contents = readfile(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sick", "__init__.py"))

version = version_regex.findall(contents)[0]

setup(name="sick",
    version=version,
    author="Andrew R. Casey",
    author_email="arc@ast.cam.ac.uk",
    packages=["sick", "sick.tests"],
    url="http://www.github.com/andycasey/sick/",
    license="MIT",
    description="Infer astrophysical parameters from spectra",
    long_description=readfile(os.path.join(os.path.dirname(__file__), "README.md")),
    install_requires=readfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt")).split("\n"),
    entry_points={
        "console_scripts": ["sick = sick.cli:main"]
    }
)

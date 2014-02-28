# coding: utf-8

""" Spectral Comparison and Parameter Evaluation """ 

import os
import re
import sys

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

major, minor1, minor2, release, serial =  sys.version_info

readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}

def readfile(filename):
    with open(filename, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents

version_regex = re.compile("__version__ = \"(.*?)\"")
contents = readfile(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scope",
    "__init__.py"))

version = version_regex.findall(contents)[0]


setup(name="scope",
      version=version,
      author="Andrew R. Casey",
      author_email="arc@ast.cam.ac.uk",
      packages=["scope"],
      url="http://www.github.com/andycasey/a-new-scope/",
      license="GPLv2",
      description="Probabilistically determine stellar parameters from spectra",
      long_description=readfile(os.path.join(os.path.dirname(__file__), "README.rst")),
      install_requires=readfile(
          os.path.join(os.path.dirname(__file__), "requirements.txt")).split("\n")
     )

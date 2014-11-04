# coding: utf-8

""" Test model class """

from __future__ import print_function

import os
import random
import string
import unittest
import yaml

import numpy as np

import sick.models as models
import sick.validation as validation

def random_string(n=10):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) \
        for _ in range(n))

def create_temporary_filename(ext=None, n=10):
    if ext is None:
        ext = ""
    filename = os.extsep.join([random_string(10), ext])
    while os.path.exists(filename):
        filename = os.extsep.join([random_string(10), ext])
    return filename    


class ModelTest(unittest.TestCase):

    def test_non_existent_channel_keys(self):

        bad_model = {
            "channels": {
                "one": {
                    "dispersion_filename": "non_existent_filename"
                }
            }
        }
        assert not os.path.exists("non_existent_filename"), "...really?"\
            " you named a file 'non_existent_filename'? who does that?!"

        model_filename = create_temporary_filename("yaml")
        with open(model_filename, "w+") as fp:
            yaml.dump(bad_model, fp)

        # Assert an IOError missing the dispersion filename
        self.assertRaises(IOError, models.Model, model_filename)

        # OK create a dispersion filename
        dispersion_filename = create_temporary_filename("txt")
        np.savetxt(dispersion_filename, np.arange(10))

        # Add the dispersion filename
        bad_model["channels"]["one"]["dispersion_filename"] = dispersion_filename
        with open(model_filename, "w") as fp:
            yaml.dump(bad_model, fp)

        # Assert a keyerror missing flux_folder
        self.assertRaises(KeyError, models.Model, model_filename)

        # Add a flux folder
        bad_model["channels"]["one"]["flux_folder"] = os.getcwd()
        with open(model_filename, "w+") as fp:
            yaml.dump(bad_model, fp)

        # Assert a KeyError missing flux_filename_match
        self.assertRaises(KeyError, models.Model, model_filename)

        # Clean up
        map(os.unlink, [dispersion_filename, model_filename])


    def runTest(self):
        pass
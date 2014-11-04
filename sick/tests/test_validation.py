# coding: utf-8

""" Test the model validation """

from __future__ import print_function

import unittest
import sick.validation as v


class UncachedModelValidationTest(unittest.TestCase):

    c = {"channels": True}
    z = ["abc", "def"]

    def test_channel_names_str(self):
        # These are normal
        self.assertTrue(v._validate_channels(self.c, ["yes", "#@$)*@#$", "ok"]))

    def test_channel_names_unicode(self):
        self.assertTrue(v._validate_channels(self.c, [u"yes", u"ok"]))

    def test_channel_with_full_stops(self):
        # Don't allow anything with a . in it.
        bad_channels = ["what.what", "not.not"]
        self.assertRaises(ValueError, v._validate_channels, 
            *[self.c, bad_channels])

    def test_channel_name_none(self):
        # Only allow strings or unicodes as names
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [None, "yes"]])

    def test_channel_name_dict(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [{}, "yes"]])

    def test_channel_name_bool1(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [False, "yes"]])

    def test_channel_name_bool2(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [True, "yes"]])

    def test_channel_name_int(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [1, "yes"]])

    def test_channel_name_float(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [2.0, "yes"]])

    def test_channel_name_tuple(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [(), "yes"]])

    def test_channel_name_list(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, [[], "yes"]])

    # Channel names must be list-like
    def test_channels_as_none(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, None])

    def test_channels_as_false(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, False])

    def test_channels_as_true(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, True])

    def test_channels_as_int(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, 1])

    def test_channels_as_float(self):
        self.assertRaises(TypeError, v._validate_channels,
            *[self.c, 2.0])
        
    # Masks are not necessary
    def test_mask_no_configuration_provided(self):
        self.assertTrue(v._validate_mask({}))

    def test_mask_as_empty_list(self):
        self.assertTrue(v._validate_mask({"mask": []}))

    def test_mask_as_empty_tuple(self):
        self.assertTrue(v._validate_mask({"mask": ()}))

    def test_mask_as_none(self):
        self.assertTrue(v._validate_mask({"mask": None}))

    def test_mask_as_false(self):
        self.assertTrue(v._validate_mask({"mask": False}))

    # When a mask is given though, it must be valid
    def test_mask_as_empty_dict(self):
        self.assertRaises(TypeError, v._validate_mask, {"mask": {}})

    def test_mask_as_bool(self):
        self.assertRaises(TypeError, v._validate_mask, {"mask": True})

    def test_mask_as_int(self):
        self.assertRaises(TypeError, v._validate_mask, {"mask": 3})

    def test_mask_as_list_with_int(self):
        self.assertRaises(TypeError, v._validate_mask, {"mask": [3]})

    def test_mask_as_non_nested_list(self):
        self.assertRaises(TypeError, v._validate_mask, {"mask": [1, 2]})

    def test_mask_as_nested_list_of_nones(self):
        self.assertRaises(TypeError, v._validate_mask, {"mask": [[None, None]]})

    def test_mask_single_region(self):
        self.assertTrue(v._validate_mask({"mask": [[1, 2]]}))

    def test_mask_multiple_regions(self):
        self.assertTrue(v._validate_mask({"mask": [[1, 2], [3, 4]]}))

    def test_mask_backward_region(self):
        # It is OK if they are not upper/lower bounded, too. But we will give
        # a warning
        self.assertTrue(v._validate_mask({"mask": [[1, 2], [4, 3]]}))

    def test_mask_equal_region(self):
        self.assertTrue(v._validate_mask({"mask": [[1, 2], [4, 4]]}))


    def test_redshift_no_configuration_provided(self):
        # Redshift is not necessary
        self.assertTrue(v._validate_redshift({}, self.z))

    def test_redshift_as_empty_list(self):
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": []}, self.z])

    def test_redshift_as_none(self):
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": None}, self.z])

    def test_redshift_as_int(self):
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": 0}, self.z])

    def test_redshift_as_float(self):
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": 1.0}, self.z])

    # When given, it can be boolean or a dict
    def test_redshift_as_empty_dict(self):
        self.assertTrue(v._validate_redshift({"redshift": {}}, self.z))

    def test_redshift_as_true(self):
        self.assertTrue(v._validate_redshift({"redshift": True}, self.z))

    def test_redshfit_as_false(self):
        self.assertTrue(v._validate_redshift({"redshift": False}, self.z))

    # We can give it for non-existant channels, too, and they will just be
    # ignored
    def test_redshift_with_non_existent_channel1(self):
        self.assertTrue(v._validate_redshift({"redshift": {"nope": True}}, self.z))

    def test_redshift_with_non_existent_channel2(self):
        self.assertTrue(v._validate_redshift({"redshift": {"nope": False}}, self.z))

    # If we don't give it to all channels, then the ones not specified will
    # default to False
    def test_redshift_with_single_existent_channel1(self):
        self.assertTrue(v._validate_redshift({"redshift": {"abc": True}}, self.z))

    def test_redshift_with_single_existent_channel2(self):
        self.assertTrue(v._validate_redshift({"redshift": {"abc": False}}, self.z))
        

    def test_convolve_no_configuration_provided(self):
        # Convolution is not necessary
        self.assertTrue(v._validate_convolve({}, self.z))

    def test_convolve_as_empty_list(self):
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": []}, self.z])

    def test_convolve_as_none(self):
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": None}, self.z])

    def test_convolve_as_int(self):
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": 0}, self.z])

    def test_convolve_as_float(self):
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": 1.0}, self.z])        

    # When given, it can be boolean or a dict
    def test_convolve_as_empty_dict(self):
        self.assertTrue(v._validate_convolve({"convolve": {}}, self.z))

    def test_convolve_as_true(self):
        self.assertTrue(v._validate_convolve({"convolve": True}, self.z))

    def test_convolve_as_false(self):
        self.assertTrue(v._validate_convolve({"convolve": False}, self.z))

    # We can give it for non-existant self.zhannels, too, and they will just be
    # ignored
    def test_convolve_with_non_existent_channel1(self):
        self.assertTrue(v._validate_convolve({"convolve": {"nope": True}}, self.z))

    def test_convolve_with_non_existent_channel2(self):
        self.assertTrue(v._validate_convolve({"convolve": {"nope": False}}, self.z))

    # If we don't give it to all channels, then the ones not specified will
    # default to False
    def test_convolve_with_single_existent_channel1(self):
        self.assertTrue(v._validate_convolve({"convolve": {"abc": True}}, self.z))

    def test_convolve_with_single_existent_channel2(self):
        self.assertTrue(v._validate_convolve({"convolve": {"abc": False}}, self.z))
        
    # Normalisation not required
    def test_normalisation_no_configuration_provided(self):
        self.assertTrue(v._validate_normalisation({}, []))

    def test_normalisation_as_false(self):
        self.assertTrue(v._validate_normalisation({"normalise": False}, []))

    # Type is important
    def test_normalisation_as_none(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": None}, []])

    def test_normalisation_as_true(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": True}, []])

    def test_normalisation_as_empty_list(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": []}, []])

    def test_normalisation_as_empty_tuple(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": ()}, []])

    def test_normalisation_as_int(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": 3}, []])

    def test_normalisation_as_float(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": 3.14}, []])

    # We can provide non-existent channels
    def test_normalisation_with_non_existent_channel(self):
        self.assertTrue(v._validate_normalisation(
            {"normalise": {"what": True}}, ["k", "yep"]))

    def test_normalisation_with_single_existent_channel_as_none(self):
        # But if we provide one that exists, the type is important
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": None}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_as_bool(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": True}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_as_empty_list(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": []}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_as_empty_tuple(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": ()}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_as_int(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": 3}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_as_float(self):
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": 3.14}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_as_empty_dict(self):
        # And if we provide one that exists, you need to provide some info on it
        self.assertRaises(KeyError, v._validate_normalisation,
            *[{"normalise": {"k": {}}}, ["k", "yep"]])

    def test_normalisation_with_single_existent_channel_with_order_as_int(self):
        # Like the order, which must be an integer-like
        self.assertTrue(v._validate_normalisation(
            {"normalise": {"k": {"order": 0}}}, ["k", "yep"]))

    def test_normalisation_with_single_existent_channel_with_order_as_float(self):
        # This will just raise a warning:
        self.assertTrue(v._validate_normalisation(
            {"normalise": {"k": {"order": 3.14}}}, ["k", "yep"]))

        # These are no good:
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": {"order": None}}}, ["k", "yep"]])
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": {"order": True}}}, ["k", "yep"]])
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": {"order": []}}}, ["k", "yep"]])
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": {"order": {}}}}, ["k", "yep"]])
        self.assertRaises(TypeError, v._validate_normalisation,
            *[{"normalise": {"k": {"order": ()}}}, ["k", "yep"]])

        # And positive
        self.assertRaises(ValueError, v._validate_normalisation,
            *[{"normalise": {"k": {"order": -1}}}, ["k", "yep"]])


    def runTest(self):
        pass

    def test_settings(self):

        p = ["z.what", "f.what"]

        self.assertRaises(KeyError, v._validate_settings, *[{}, p])
        
        # But it does have to be the right type
        self.assertRaises(TypeError, v._validate_settings, *[{"settings": None}, ])
        self.assertRaises(TypeError, v._validate_settings, *[{"settings": 1}, ])
        self.assertRaises(TypeError, v._validate_settings, *[{"settings": False}, ])
        self.assertRaises(TypeError, v._validate_settings, *[{"settings": []}, ])
        self.assertRaises(TypeError, v._validate_settings, *[{"settings": 3.14}, ])
        
        # Check missing values
        config = {"settings": {"walkers": 4}}
        self.assertRaises(KeyError, v._validate_settings, *[config, p])

        config = {"settings": {"walkers": 4, "burn": 10}}
        self.assertRaises(KeyError, v._validate_settings, *[config, p])
        
        config = {"settings": {"walkers": 4, "burn": 10, "sample": 10}}
        self.assertRaises(KeyError, v._validate_settings, *[config, p])
        
        # Check walker numbers
        config = {"settings": {"walkers": 5, "burn": 10, "sample": 10, "threads": 1}}
        self.assertRaises(ValueError, v._validate_settings, *[config, p])        

        config = {"settings": {"walkers": 2, "burn": 10, "sample": 10, "threads": 1}}
        self.assertRaises(ValueError, v._validate_settings, *[config, p])
        
        config = {"settings": {"walkers": 2 * len(p), "burn": 10, "sample": 5, "threads": 1}}
        self.assertTrue(v._validate_settings(config, p))

        # Check the burn/sample/walker numbers
        bad_types = (None, True, False, [], {})
        for bad_type in bad_types:

            ok = {"settings": {"walkers": 10, "sample": 10, "burn": 10}}

            config = ok.copy()
            config["settings"]["walkers"] = bad_type
            self.assertRaises(TypeError, v._validate_settings, *[config, p])
            
            config = ok.copy()
            config["settings"]["burn"] = bad_type
            self.assertRaises(TypeError, v._validate_settings, *[config, p])

            config = ok.copy()
            config["settings"]["sample"] = bad_type
            self.assertRaises(TypeError, v._validate_settings, *[config, p])

            config = ok.copy()
            config["settings"]["threads"] = bad_type
            self.assertRaises(TypeError, v._validate_settings, *[config, p])

        # All values should be positive
        config = {"settings": {"burn": -1, "sample": 10, "walkers": 10, "threads": 1}}
        self.assertRaises(ValueError, v._validate_settings, *[config, p])

        config = {"settings": {"sample": -1, "burn": 10, "walkers": 10, "threads": 1}}
        self.assertRaises(ValueError, v._validate_settings, *[config, p])

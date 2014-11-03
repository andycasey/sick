# coding: utf-8

""" Test the model validation """

from __future__ import print_function

import unittest
import sick.validation as v


class UncachedModelValidationTest(unittest.TestCase):

    def test_channel_names(self):

        c = {"channels": True}

        # These are normal
        self.assertTrue(v._validate_channels, [c, ["yes", "#@$)(&*@#$", "ok"], []])
        self.assertTrue(v._validate_channels, [c, [u"yes", u"ok"], []])

        # Don't allow anything with a . in it.
        bad_channels = ["what.what", "not.not"]
        self.assertRaises(ValueError, v._validate_channels, 
            *[c, bad_channels])

        # Only allow strings or unicodes as names
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [None, "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [{}, "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [False, "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [True, "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [1, "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [2.0, "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [(), "yes"]])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, [[], "yes"]])

        # Channel names must be list-like
        self.assertRaises(TypeError, v._validate_channels,
            *[c, None])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, False])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, True])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, 1])
        self.assertRaises(TypeError, v._validate_channels,
            *[c, 2.0])
        

    def test_mask(self):

        # Masks are not necessary
        self.assertTrue(v._validate_mask, {})
        self.assertTrue(v._validate_mask, {"mask": []})
        self.assertTrue(v._validate_mask, {"mask": ()})
        self.assertTrue(v._validate_mask, {"mask": None})
        self.assertTrue(v._validate_mask, {"mask": False})

        # When a mask is given though, it must be valid
        self.assertRaises(TypeError, v._validate_mask, {"mask": {}})
        self.assertRaises(TypeError, v._validate_mask, {"mask": True})
        self.assertRaises(TypeError, v._validate_mask, {"mask": 3})
        self.assertRaises(TypeError, v._validate_mask, {"mask": [3]})
        self.assertRaises(TypeError, v._validate_mask, {"mask": [1, 2]})
        self.assertRaises(TypeError, v._validate_mask, {"mask": [[None, None]]})

        # These are OK:
        self.assertTrue(v._validate_mask, {"mask": [[1, 2]]})
        self.assertTrue(v._validate_mask, {"mask": [[1, 2], [3, 4]]})

        # It is OK if they are not upper/lower bounded, too. But we will give
        # a warning
        self.assertTrue(v._validate_mask, {"mask": [[1, 2], [4, 3]]})

    def test_mask2(self):
        # OK if they are the same point too, but that makes no sense in reality
        print("OK WE ARE DOING IT\n\n\n\n\n\n\n\n\n")
        self.assertTrue(v._validate_mask, {"mask": [[1, 2], [4, 4]]})


    def test_redshift(self):

        c = ["abc", "def"]

        # Redshift is not necessary
        self.assertTrue(v._validate_redshift, [{}, c])

        # Must be of correct type (most are)
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": []}, c])
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": None}, c])
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": 0}, c])
        self.assertRaises(TypeError, v._validate_redshift, *[{"redshift": 1.0}, c])        

        # When given, it can be boolean or a dict
        self.assertTrue(v._validate_redshift, [{"redshift": {}}, c])
        self.assertTrue(v._validate_redshift, [{"redshift": True}, c])
        self.assertTrue(v._validate_redshift, [{"redshift": False}, c])

        # We can give it for non-existant channels, too, and they will just be
        # ignored
        self.assertTrue(v._validate_redshift, [{"redshift": {"nope": True}}, c])
        self.assertTrue(v._validate_redshift, [{"redshift": {"nope": False}}, c])

        # If we don't give it to all channels, then the ones not specified will
        # default to False
        self.assertTrue(v._validate_redshift, [{"redshift": {"abc": True}}, c])
        self.assertTrue(v._validate_redshift, [{"redshift": {"abc": False}}, c])
        

    def test_convolve(self):

        c = ["abc", "def"]

        # Convolution is not necessary
        self.assertTrue(v._validate_convolve, [{}, c])

        # Must be of correct type (most are)
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": []}, c])
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": None}, c])
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": 0}, c])
        self.assertRaises(TypeError, v._validate_convolve, *[{"convolve": 1.0}, c])        

        # When given, it can be boolean or a dict
        self.assertTrue(v._validate_convolve, [{"convolve": {}}, c])
        self.assertTrue(v._validate_convolve, [{"convolve": True}, c])
        self.assertTrue(v._validate_convolve, [{"convolve": False}, c])

        # We can give it for non-existant channels, too, and they will just be
        # ignored
        self.assertTrue(v._validate_convolve, [{"convolve": {"nope": True}}, c])
        self.assertTrue(v._validate_convolve, [{"convolve": {"nope": False}}, c])

        # If we don't give it to all channels, then the ones not specified will
        # default to False
        self.assertTrue(v._validate_convolve, [{"convolve": {"abc": True}}, c])
        self.assertTrue(v._validate_convolve, [{"convolve": {"abc": False}}, c])
        

    def test_settings(self):

        p = ["z.what", "f.what"]

        # You don't need ... anything
        self.assertTrue(v._validate_settings, [{}, p])
        
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
        self.assertTrue(v._validate_settings, [config, p])

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


    def test_normalisation(self):

        f = v._validate_normalisation

        # Normalisation not required
        self.assertTrue(f, [{}, None])
        self.assertTrue(f, [{"normalise": False}, None])

        # Type is important
        self.assertRaises(TypeError, f, *[{"normalise": None}, None])
        self.assertRaises(TypeError, f, *[{"normalise": True}, None])
        self.assertRaises(TypeError, f, *[{"normalise": []}, None])
        self.assertRaises(TypeError, f, *[{"normalise": ()}, None])
        self.assertRaises(TypeError, f, *[{"normalise": 3}, None])
        self.assertRaises(TypeError, f, *[{"normalise": 3.14}, None])

        # We can provide non-existent channels
        c = ["k", "yep"]
        self.assertTrue(f, [{"normalise": {"what": True}}, c])

        # But if we provide one that exists, the type is important
        self.assertRaises(TypeError, f, *[{"normalise": {"k": None}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": True}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": []}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": ()}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": 3}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": 3.14}}, c])

        # And if we provide one that exists, you need to provide some info on it
        self.assertRaises(KeyError, f, *[{"normalise": {"k": {}}}, c])

        # Like the order, which must be an integer-like
        self.assertTrue(f, [{"normalise": {"k": {"order": 0}}}, c])
        # This will just raise a warning:
        self.assertTrue(f, [{"normalise": {"k": {"order": 3.14}}}, c])

        # These are no good:
        self.assertRaises(TypeError, f, *[{"normalise": {"k": {"order": None}}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": {"order": True}}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": {"order": []}}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": {"order": {}}}}, c])
        self.assertRaises(TypeError, f, *[{"normalise": {"k": {"order": ()}}}, c])

        # And positive
        self.assertRaises(ValueError, f, *[{"normalise": {"k": {"order": -1}}}, c])
        
        # We should probably require a method specification too, but not now

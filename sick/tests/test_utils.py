# coding: utf-8

""" Test sick utilities """

from __future__ import print_function

import numpy as np
from time import time, sleep

import sick.utils as utils

def test_latexify():

    input_labels = ("teff", "f.blue", "v.blue", "normalise.red.a1")
    output_labels = ['$T_{\\rm eff}$ [K]', '$ln(f_{blue})$', '$V_{rad,{blue}}$ [km/s]', '$r_{1}$']
    assert utils.latexify(input_labels) == output_labels
    assert utils.latexify(input_labels[0]) == output_labels[0]
    assert utils.latexify("convolve.green") == "$\sigma_{green}$ [$\AA$]"
    assert utils.latexify(["this", "moo"], {"this": "$that$"}) == ["$that$", "moo"]


def test_unique_preserved_list():

    data = ("this", "this", "that", 3, 4, 5, 3)

    assert utils.unique_preserved_list(data) == ["this", "that", 3, 4, 5]
    assert utils.unique_preserved_list([]) == []
    assert utils.unique_preserved_list([5]) == [5]


def test_human_readable_digit():

    func = lambda x: utils.human_readable_digit(x).rstrip()
    assert func(3000) == "3.0 thousand"
    assert func(5) == "5.0"
    assert func(100) == "100.0"
    assert func(0) == "0.0"
    assert func(1340000) == "1.3 million"
    assert func(7.2e9) == "7.2 billion"


def test_wrapper():

    func = lambda x, y, z: x**2 + y**3 - z
    func_wrap = utils.wrapper(func, [5, 3])
    assert func_wrap(1.23) == 123.5129


def test_lru_cacher():

    @utils.lru_cache(maxsize=10)
    def func(x):
        sleep(5)
        return x**2

    t_init = time()
    x = np.random.uniform(-1000, 1000)
    y1 = func(x)
    t_a = time() - t_init

    t_init = time()
    y2 = func(x)
    t_b = time() - t_init

    assert y2 == y1
    assert (t_a - t_b) > 4 # assumes func(x) takes less than 1 second
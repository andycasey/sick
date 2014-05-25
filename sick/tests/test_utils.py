# coding: utf-8

""" Test sick utilities """

from __future__ import print_function

import sick.utils as utils

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


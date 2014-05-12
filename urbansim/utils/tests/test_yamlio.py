import os

from .. import yamlio


def test_ordered_yaml():
    d = {
        'name': 'test',
        'ytransform': 'xyz',
        'unordered': 'abc'
    }

    test_yaml = yamlio.ordered_yaml(d)

    expected_yaml = (
        'name: test{linesep}'
        'ytransform: xyz{linesep}'
        'unordered: abc{linesep}').format(linesep=os.linesep)

    assert test_yaml == expected_yaml

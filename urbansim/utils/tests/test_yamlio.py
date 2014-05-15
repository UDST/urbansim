import os
import tempfile
from StringIO import StringIO

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.util import testing as pdt

from .. import yamlio


@pytest.fixture
def test_cfg():
    return {
        'name': 'test',
        'ytransform': 'xyz',
        'unordered': 'abc'
    }


@pytest.fixture
def expected_yaml():
    return (
        'name: test{linesep}'
        'ytransform: xyz{linesep}'
        'unordered: abc{linesep}').format(linesep=os.linesep)


@pytest.fixture
def test_file(request):
    name = tempfile.NamedTemporaryFile(suffix='.yaml').name

    def cleanup():
        if os.path.exists(name):
            os.remove(name)
    request.addfinalizer(cleanup)

    return name


def test_ordered_yaml(test_cfg, expected_yaml):
    test_yaml = yamlio.ordered_yaml(test_cfg)
    assert test_yaml == expected_yaml


def test_convert_to_yaml_str(test_cfg, expected_yaml):
    test_yaml = yamlio.convert_to_yaml(test_cfg, str_or_buffer=None)
    assert test_yaml == expected_yaml


def test_convert_to_yaml_file(test_cfg, expected_yaml, test_file):
    yamlio.convert_to_yaml(test_cfg, test_file)

    with open(test_file) as f:
        assert f.read() == expected_yaml


def test_convert_to_yaml_buffer(test_cfg, expected_yaml):
    test_buffer = StringIO()
    yamlio.convert_to_yaml(test_cfg, test_buffer)

    assert test_buffer.getvalue() == expected_yaml


class Test_yaml_to_dict(object):
    @classmethod
    def setup_class(cls):
        cls.yaml_str = """
a:
  x: 1
  y: 2
  z: 3
b:
  x: 3
  y: 4
  z: 5
"""
        cls.expect_dict = {
            'a': {'x': 1, 'y': 2, 'z': 3},
            'b': {'x': 3, 'y': 4, 'z': 5}}

    def test_str(self):
        assert yamlio.yaml_to_dict(yaml_str=self.yaml_str) == self.expect_dict

    def test_file(self, test_file):
        with open(test_file, 'w') as f:
            f.write(self.yaml_str)

        assert yamlio.yaml_to_dict(str_or_buffer=test_file) == self.expect_dict

    def test_buffer(self):
        buff = StringIO(self.yaml_str)
        buff.seek(0)

        assert yamlio.yaml_to_dict(str_or_buffer=buff) == self.expect_dict

    def test_raises(self):
        with pytest.raises(ValueError):
            yamlio.yaml_to_dict()


def test_series_to_yaml_safe_int_index():
    s = pd.Series(np.arange(100, 103), index=np.arange(3))
    d = yamlio.series_to_yaml_safe(s)

    assert d == {0: 100, 1: 101, 2: 102}
    y = yaml.dump(d, default_flow_style=False)
    pdt.assert_series_equal(pd.Series(yaml.load(y)), s)


def test_series_to_yaml_safe_str_index():
    s = pd.Series(
        np.array(['a', 'b', 'c']), index=np.array(['x', 'y', 'z']))
    d = yamlio.series_to_yaml_safe(s)

    assert d == {'x': 'a', 'y': 'b', 'z': 'c'}
    y = yaml.dump(d, default_flow_style=False)
    pdt.assert_series_equal(pd.Series(yaml.load(y)), s)


def test_frame_to_yaml_safe():
    df = pd.DataFrame(
        {'col1': np.array([100, 200, 300]),
         'col2': np.array(['a', 'b', 'c'])},
        index=np.arange(3))
    d = yamlio.frame_to_yaml_safe(df)

    assert d == {'col1': {0: 100, 1: 200, 2: 300},
                 'col2': {0: 'a', 1: 'b', 2: 'c'}}
    y = yaml.dump(d, default_flow_style=False)
    pdt.assert_frame_equal(pd.DataFrame(yaml.load(y)), df)

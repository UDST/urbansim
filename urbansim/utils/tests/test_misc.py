import pytest

from .. import misc


class _FakeTable(object):
    def __init__(self, name, columns):
        self.name = name
        self.columns = columns


@pytest.fixture
def fta():
    return _FakeTable('a', ['aa', 'ab', 'ac'])


@pytest.fixture
def ftb():
    return _FakeTable('b', ['bx', 'by', 'bz'])


def test_column_map_raises(fta, ftb):
    with pytest.raises(RuntimeError):
        misc.column_map([fta, ftb], ['aa', 'by', 'bz', 'cw'])


def test_column_map_none(fta, ftb):
    assert misc.column_map([fta, ftb], None) == {'a': None, 'b': None}


def test_column_map(fta, ftb):
    assert misc.column_map([fta, ftb], ['aa', 'by', 'bz']) == \
        {'a': ['aa'], 'b': ['by', 'bz']}
    assert misc.column_map([fta, ftb], ['by', 'bz']) == \
        {'a': [], 'b': ['by', 'bz']}

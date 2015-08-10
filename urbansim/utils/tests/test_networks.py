import os.path
import tempfile

import numpy as np
import orca
import pandana as pdna
import pandas as pd
import pytest

from .. import networks


@pytest.fixture(scope="module")
def sample_osm(request):
    store = pd.HDFStore(
        os.path.join(os.path.dirname(__file__), 'osm_sample.h5'), "r")
    nodes, edges = store.nodes, store.edges
    net = pdna.Network(nodes.x, nodes.y, edges["from"], edges.to,
                       edges[["weight"]])

    net.precompute(500)

    def fin():
        store.close()
    request.addfinalizer(fin)

    return net


@pytest.fixture
def test_file(request):
    name = tempfile.NamedTemporaryFile(suffix='.yaml').name

    def cleanup():
        if os.path.exists(name):
            os.remove(name)
    request.addfinalizer(cleanup)

    return name


@pytest.fixture()
def sample_df(sample_osm):
    num_rows = 500
    index = np.random.choice(sample_osm.node_ids, num_rows)
    df = pd.DataFrame({"test_col_name": np.random.random(num_rows),
                       "_node_id": index})
    return df


def test_networks_yaml(sample_osm, sample_df, test_file):

    @orca.table('testing_df', cache=True)
    def source():
        return sample_df

    s = """
name: networks

desc: Neighborhood Accessibility Variables

model_type: networks

node_col: _node_id

variable_definitions:

  - name: test_attr
    dataframe: testing_df
    varname: test_col_name
    radius: 500
    apply: np.log1p
    filters:
    - test_col_name > .1
    """

    f = open(test_file, "w")
    f.write(s)
    f.close()

    df = networks.from_yaml(sample_osm, test_file)

    assert len(df) == 1498
    assert df.describe()['test_attr']['max'] > 0
    assert df.describe()['test_attr']['min'] == 0
    assert df.describe()['test_attr']['std'] > 0
    ind = pd.Series(df.index).describe()
    assert ind.loc['min'] > 0
    assert ind.loc['count'] == 1498

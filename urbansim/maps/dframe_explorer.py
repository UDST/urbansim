from bottle import route, response, run, hook, static_file
from urbansim.utils import yamlio
import simplejson
import numpy as np
import pandas as pd
import os
from jinja2 import Environment


@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = \
        'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

DFRAMES = {}
CONFIG = None


def get_schema():
    global DFRAMES
    return {name: list(DFRAMES[name].columns) for name in DFRAMES}


@route('/map_query/<table>/<filter>/<groupby>/<field:path>/<agg>', method="GET")
def map_query(table, filter, groupby, field, agg):
    global DFRAMES

    filter = ".query('%s')" % filter if filter != "empty" else ""

    df = DFRAMES[table]

    if field not in df.columns:
        print "Col not found, trying eval:", field
        df["eval"] = df.eval(field)
        field = "eval"

    cmd = "df%s.groupby('%s')['%s'].%s" % \
          (filter, groupby, field, agg)
    print cmd
    results = eval(cmd)
    results[results == np.inf] = np.nan
    results = yamlio.series_to_yaml_safe(results.dropna())
    results = {int(k): results[k] for k in results}
    return results


@route('/map_query/<table>/<filter>/<groupby>/<field>/<agg>', method="OPTIONS")
def ans_options(table=None, filter=None, groupby=None, field=None, agg=None):
    return {}


@route('/')
def index():
    global CONFIG
    dir = os.path.dirname(__file__)
    index = open(os.path.join(dir, 'dframe_explorer.html')).read()
    return Environment().from_string(index).render(CONFIG)


@route('/data/<filename>')
def data_static(filename):
    return static_file(filename, root='./data')


def start(views,
          center=[37.7792, -122.2191],
          zoom=11,
          shape_json='data/zones.json',
          geom_name='ZONE_ID',  # from JSON file
          join_name='zone_id',  # from data frames
          precision=8,
          port=8765,
          host='localhost',
          testing=False):

    global DFRAMES, CONFIG
    DFRAMES = {str(k): views[k] for k in views}

    config = {
        'center': str(center),
        'zoom': zoom,
        'shape_json': shape_json,
        'geom_name': geom_name,
        'join_name': join_name,
        'precision': precision
    }

    for k in views:
        if join_name not in views[k].columns:
            raise Exception("Join name must be present on all dataframes - "
                            "'%s' not present on '%s'" % (join_name, k))

    config['schema'] = simplejson.dumps(get_schema())

    CONFIG = config

    if testing:
        return

    run(host=host, port=port, debug=True)

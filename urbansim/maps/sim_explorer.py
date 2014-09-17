from bottle import route, response, run, hook, static_file
import os
from jinja2 import Environment
import webbrowser


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


@route('/map_query/<table>/<filter>/<groupby>/<field>/<agg>', method="OPTIONS")
def ans_options(table=None, filter=None, groupby=None, field=None, agg=None):
    return {}


@route('/')
def index():
    global CONFIG
    dir = os.path.dirname(__file__)
    index = open(os.path.join(dir, 'sim_explorer.html')).read()
    return Environment().from_string(index).render(CONFIG)


@route('/data/<filename>')
def data_static(filename):
    return static_file(filename, root='./data')


def start(sim_output,
          center=[37.7792, -122.2191],
          zoom=11,
          shape_json='data/zones.json',
          geom_name='ZONE_ID',  # from JSON file
          precision=2,
          port=8765,
          host='localhost',
          testing=False,
          write_static_file=None):
    """
    Start the web service which serves the Pandas queries and generates the
    HTML for the map.  You will need to open a web browser and navigate to
    http://localhost:8765 (or the specified port)

    Parameters
    ----------
    sim_output : filename
        The json output from a simulation run
    center : a Python list with two floats
        The initial latitude and longitude of the center of the map
    zoom : int
        The initial zoom level of the map
    shape_json : str
        The path to the geojson file which contains that shapes that will be
        displayed
    geom_name : str
        The field name from the JSON file which contains the id of the geometry
    precision : int
        The precision of values to show in the legend on the map
    port : int
        The port for the web service to respond on
    host : str
        The hostname to run the web service from
    testing : bool
        Whether to print extra debug information
    write_static_file : string
        Write to the given file name instead of starting the server (for
        viewing at a later time)

    Returns
    -------
    Does not return - takes over control of the thread and responds to
    queries from a web browser
    """

    global CONFIG

    CONFIG = {
        'sim_output': open(sim_output).read(),
        'center': str(center),
        'zoom': zoom,
        'shape_json': shape_json,
        'geom_name': geom_name,
        'precision': precision
    }

    if testing:
        return

    if write_static_file is not None:
        open(write_static_file, "w").write(index())
        return

    webbrowser.open("http://%s:%s" % (host, port), new=2)

    run(host=host, port=port, debug=True)

'''
TODO
add internal indicators (prices, dev, and such)
add parcels development info

d3 transitions on the shapes
play forever in a loop (randomize inputs)

pass in a second scenario to do comparison
meta information
counties/cities (mapping of zones to counties/cities)
add another shape (e.g. transit lines)
county to county flows
'''

"""to run:
$ export PYTHONPATH=path/to/urbansim
$ cd path/to/bayarea_urbansim
$ celery worker -A urbansim.server.tasks -l info

flower monitor needed to keep track of finished tasks
$ celery flower -A urbansim.server.tasks
"""

import cPickle
import decimal
import json
import math
import os
import string
import sys
import time
from decimal import Decimal
import traceback
import time

import numpy
import pandas as pd
import simplejson

from urbansim.urbansim import modelcompile
from urbansim.utils import misc

from celery import Celery

sys.path.insert(0, ".")
from bayarea import dataset

app = Celery()
app.config_from_object({
    'BROKER_URL': 'redis://localhost:6379/0',
    'CELERY_RESULT_BACKEND': 'redis://localhost:6379/0',
    'CELERYD_POOL_RESTARTS': True,  # Required for /worker/pool/restart API
    'CELERY_TRACK_STARTED': True,
})

# should we do this on each task?
DSET = dataset.LocalDataset('../bayarea_urbansim/data/bayarea.h5')


def jsonp(request, dictionary):
    if (request.query.callback):
        return "%s(%s)" % (request.query.callback, dictionary)
    return dictionary


@app.task(bind=True)
def execmodel(self, req_query):
    req = simplejson.loads(req_query.json)
    meta = app.AsyncResult(self.request.id)._get_task_meta()["result"]
    meta = {"status_msg": ["Started"]}
    if req["model"] == "modelset":
        meta["progress_count"] = 0
        years = req.get("numyearstorun", 1)
        models = len(req["modelstorun"])
        meta["progress_total"] = years * models
    self.update_state(meta=meta)

    def resp(modelname, mode, celery_task):
        print "Request: %s\n" % req_query.json
        # req = simplejson.loads(req_query.json)
        returnobj = modelcompile.run_model(
            req, DSET, configname=modelname, mode=mode, celery_task=celery_task)
        return returnobj
    modelname = req_query.get('modelname', 'autorun')
    mode = req_query.get('mode', 'estimate')
    return resp(modelname, mode, self)


@app.task(bind=True)
def get_chart_data(self, req):
    # self.update_state(state='CUSTOM_STATE')
    with self.app.events.default_dispatcher() as dispatcher:
        print "here comes dispatcher"
        print dispatcher
        dispatcher.send('task-custom_state', state='mi primer estado custom')
    # time.sleep(30)

    DSET = dataset.LocalDataset('../bayarea_urbansim/data/bayarea.h5')
    table = req.get('table', '')
    metric = req.get('metric', '')
    groupby = req.get('groupby', '')
    sort = req.get('sort', '')
    limit = req.get('limit', '')
    where = req.get('filter', '')
    orderdesc = req.get('orderdesc', '')
    jointobuildings = req.get('jointobuildings', False)
    chart_type = req.get('chart', False)

    if where:
        where = "[DSET.fetch('%s').apply(lambda x: bool(%s),axis=1)]" % (
            table, where)
    else:
        where = ""
    if sort and orderdesc:
        sort = ".order(ascending=False)"
    if sort and not orderdesc:
        sort = ".order(ascending=True)"
    if not sort and orderdesc:
        sort = ".sort_index(ascending=False)"
    if not sort and not orderdesc:
        sort = ".sort_index(ascending=True)"
    if limit:
        limit = ".head(%s)" % limit
    else:
        limit = ""

    s = "DSET.fetch('%s',addzoneid=%s)%s" % (table, str(jointobuildings), where)

    s = s + ".groupby('%s').%s%s%s" % (groupby, metric, sort, limit)

    print "Executing %s\n" % s
    recs = eval(s)

    # if 'key_dictionary' in req:
    #     key_dictionary = req['key_dictionary']
    #     # not sure /configs is the proper place to save dicts
    #     dictionary_file = open("configs/" + key_dictionary).read()
    #     dictionary = json.loads(dictionary_file)
    #     # attention: the dictionary has keys from 0 to 15, ids come from 0 to 16
    #     recs = [[dictionary[str(int(x))], float(recs.ix[x]) / 1000]
    #             for x in recs.index]
    # else:
    #     recs = [[x, float(recs.ix[x]) / 1000] for x in recs.index]

    if chart_type == 'piechart':
        return [[x, recs.ix[x]] for x in recs.index]
    else:  # works for multi-bar-charts
        recs = [[x, float(recs.ix[x]) / 1000] for x in recs.index]
        return [{'key': '', 'values': recs}]


@app.task()
def report_item_data(item):
    # let's see if it prevents pandas errors
    DSET = dataset.LocalDataset('../bayarea_urbansim/data/bayarea.h5')
    recs = None
    config = None
    template = None

    def isChart(i):
        return "chart" in i

    if isChart(item):
        config = open(os.path.join(misc.charts_dir(), item)).read()
        config = json.loads(config)

        def chart_type(c):
            return "bar-chart"

        if chart_type(item) == "bar-chart":
            recs = get_chart_data(config)
            template = """
                <h2 style="text-align: center;">%s</h2>
                <nvd3-multi-bar-chart
                        data="report_data['%s'].data"
                        id="%s"
                        height="500"
                        margin="{top: 10, right: 10, bottom: 50 , left: 80}"
                        interactive="true"
                        tooltips="true"
                        showxaxis="true"
                        xaxislabel="%s"
                        yaxislabel="%s in thousands"
                        showyaxis="true"
                        xaxisrotatelabels="0"
                        nodata="an error occurred in the chart"
                        >
                    <svg></svg>
                </nvd3-multi-bar-chart>
                        """ % (config['desc'], item, item[:-5],
                               config['groupby'], config['metric'])
            # ids wouldn't not work without the [:-5]
    else:   # map
        config = open(os.path.join(misc.maps_dir(), item)).read()
        config = json.loads(config)
        recs = get_chart_data(config)
        template = """
            <h2 style="text-align: center;">%s</h2>
            <div id="%s" mapdirective style="height: 500px; width: 100%%;">
            </div>
                   """ % (config['desc'], item)

    # s = simplejson.dumps(
    #     {'template': template, 'data': [{'key': '', 'values': recs}],
    #         'config': config},
    #     use_decimal=True
    #     )
    s = {'template': template, 'data': [{'key': '', 'values': recs}], 'config': config}
    print "response: %s\n" % s
    # return jsonp(request, s)
    return s

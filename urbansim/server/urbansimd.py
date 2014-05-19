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
import httplib2
import yaml

import numpy
import pandas as pd
import simplejson
from bottle import Bottle, route, run, response, hook, request, post, error

from urbansim.urbansim import modelcompile
from urbansim.utils import misc

from urbansim.server import tasks


def jsonp(request, dictionary):
    if (request.query.callback):
        return "%s(%s)" % (request.query.callback, dictionary)
    return dictionary


def encode_float32(obj):
    if isinstance(obj, numpy.float32) or isinstance(obj, numpy.float64):
        return float(obj)
    if isinstance(obj, numpy.int32) or isinstance(obj, numpy.int64):
        return int(obj)
    if isinstance(obj, numpy.bool_):
        return bool(obj)
    print type(obj)
    raise TypeError(repr(obj) + " is not JSON serializable")


def wrap_request(request, response, obj):
    if request.query.callback:
        response.content_type = "application/javascript"
    s = simplejson.dumps(obj, ignore_nan=True, default=encode_float32)
    print "Response: %s\n" % s
    return jsonp(request, s)


def catch_exceptions(func):
    def inner(*args, **kwargs):
        def resp():
            try:
                return {"success": func(*args, **kwargs)}
            except:
                e = sys.exc_info()
                return {"error": {"type": str(e[0]), "value": str(e[1]),
                        "traceback": traceback.format_exc()}}
        return wrap_request(request, response, resp())
    return inner


def poll_result(task_id):
    res = tasks.app.AsyncResult(task_id)
    print res.state
    ready = res.ready()
    if ready:
        data = res.get()
    else:
        data = res.state.split(",")
    return {"ready": ready, "data": data}


@hook('after_request')
def enable_cors():
    """
    You need to add some headers to each request.
    Don't use the wildcard '*' for Access-Control-Allow-Origin in production.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = \
        'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = \
        'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


@route('/configs')
def list_configs():
    def resp():
        model_type = request.query.get('typeOfModel')
        files = [f for f in os.listdir(misc.configs_dir())
                 if f[-5:] == '.yaml']
        return files

        def is_modelset(f):
            c = open(os.path.join(misc.configs_dir(), f)).read()
            c = json.loads(c)
            return 'model' in c and c['model'] == 'modelset'
        # what to do when just created models/sims, with no 'model' attr?
        if model_type == 'modelset':
            return [f for f in files if is_modelset(f)]
        else:
            return [f for f in files if not(is_modelset(f))]
    return wrap_request(request, response, resp())


# make it REST. should be /config
@route('/config_new/<configname>', method="PUT")
def new_config(configname):
    config = "{}"
    f = open(os.path.join(misc.configs_dir(), configname), "w")
    f.write(config)
    f.close()


@route('/config/<configname>', method="GET")
def read_config(configname):
    def resp():
        c = open(os.path.join(misc.configs_dir(), configname))
        return yaml.load(c)
    return wrap_request(request, response, resp())


# figure out yaml.dump parameters for the desired format
@route('/config/<configname>', method="PUT")
def write_config(configname):
    json = request.json

    def resp():
        s = simplejson.dumps(json, indent=4)
        print s
        return open(os.path.join(misc.configs_dir(), configname), "w").write(s)
    return wrap_request(request, response, resp())


@route('/charts', method="GET")
def list_charts():
    def resp():
        files = os.listdir(misc.charts_dir())
        return files
    return wrap_request(request, response, resp())


@route('/chart/<chartname>', method="GET")
def read_config(chartname):
    def resp():
        c = open(os.path.join(misc.charts_dir(), chartname))
        return yaml.load(c)
    return wrap_request(request, response, resp())


@route('/chart/<chartname>', method="PUT")
def write_config(chartname):
    json = request.json

    def resp():
        s = simplejson.dumps(json, indent=4)
        print s
        return open(os.path.join(misc.charts_dir(), chartname), "w").write(s)
    return wrap_request(request, response, resp())


# figure out proper REST routes
@route('/chartdata', method=['OPTIONS', 'GET', 'POST'])
def query():
    if request.method == "OPTIONS":
        return {}
    req = request.query.json
    if (request.query.callback):
        response.content_type = "application/javascript"
    print "Request: %s\n" % request.query.json
    req = simplejson.loads(req)
    # recs = tasks.get_chart_data.delay(req).get()
    # recs = get_chart_data(req)
    # s = simplejson.dumps([{'key': '', 'values': recs}], use_decimal=True)
    # print "Response: %s\n" % s
    # return jsonp(request, s)
    # return [{'key': '', 'values': recs}]
    task_id = tasks.get_chart_data.delay(req).id
    print task_id
    return task_id


@route('/chartdata/<task_id>', method=['OPTIONS', 'GET', 'POST'])
@catch_exceptions
def ask_result(task_id):
    return poll_result(task_id)


@route('/maps', method="GET")
def list_maps():
    def resp():
        files = os.listdir(misc.maps_dir())
        return files
    return wrap_request(request, response, resp())


@route('/map/<mapname>', method="GET")
def read_config(mapname):
    def resp():
        c = open(os.path.join(misc.maps_dir(), mapname))
        return yaml.loads(c)
    return wrap_request(request, response, resp())


@route('/map/<mapname>', method="PUT")
def write_config(mapname):
    json = request.json

    def resp():
        s = simplejson.dumps(json, indent=4)
        print s
        return open(os.path.join(misc.maps_dir(), mapname), "w").write(s)
    return wrap_request(request, response, resp())


@route('/reports', method="GET")
def list_reports():
    def resp():
        files = os.listdir(misc.reports_dir())
        return files
    return wrap_request(request, response, resp())


@route('/report/<reportname>', method="GET")
def read_config(reportname):
    def resp():
        c = open(os.path.join(misc.reports_dir(), reportname))
        return yaml.load(c)
    return wrap_request(request, response, resp())


@route('/report/<reportname>', method="PUT")
def write_config(reportname):
    json = request.json

    def resp():
        s = simplejson.dumps(json, indent=4)
        print s
        return open(os.path.join(misc.reports_dir(), reportname), "w").write(s)
    return wrap_request(request, response, resp())


# should all this function be a celery task, or just get_chart-data?
@route('/report_data_task/<item>', method="GET")
def call_task(item):
    return tasks.report_item_data.delay(item).id


@route('/report_data_result/<task_id>', method="GET")
@catch_exceptions
def ask_result(task_id):
    return poll_result(task_id)


@route('/datasets')
def list_datasets():
    def resp():
        return DSET.list_tbls()
    return wrap_request(request, response, resp())


@route('/datasets/<name>/columns')
def columns_dataset(name):
    def resp():
        return DSET.fetch(name).columns.tolist()
    return wrap_request(request, response, resp())


@route('/dataset_read')
def dataset_read():
    def resp():
        url = request.query.get('url', None)
        storename = request.query.get('outname', None)
        DSET.save_tmptbl(storename, pd.read_csv(url))
        return DSET.list_tbls()
    return wrap_request(request, response, resp())


@route('/merge_datasets/<leftname>/<rightname>/<lefton>/<righton>/<how>/<outname>')
def merge_datasets(leftname, rightname, lefton, righton, how, outname):
    def resp(leftname, rightname, lefton, righton, how, outname):
        left_index = right_index = False
        if lefton == "index":
            lefton = None
            left_index = True
        if righton == "index":
            righton = None
            right_index = True
        left = DSET.fetch(leftname)
        right = DSET.fetch(rightname)
        df = pd.merge(left, right, left_on=lefton, right_on=righton,
                      left_index=left_index, right_index=right_index, how=how)
        DSET.save_tmptbl(outname, df)
        return DSET.list_tbls()

    return wrap_request(
        request, response,
        resp(leftname, rightname, lefton, righton, how, outname))


@route('/datasets/<name>/info')
def datasets_info(name):
    def resp(name):
        df = DSET.fetch(name)
        return {"schema": [{"label": x, "simpletype": str(df.dtypes.ix[x]),
                            "cardinality": int(df[x].count())}
                           for x in DSET.fetch(name).columns]}
    return wrap_request(request, response, resp(name))


@route('/datasets/<name>/summary')
def datasets_summary(name):
    def resp(name):
        df = DSET.fetch(name).describe().transpose()
        d = {k: {"summary": {i: float(v.ix[i]) for i in v.index}}
             for k, v in df.iterrows()}
        df = DSET.fetch(name)
        for col in df.columns:
            d.setdefault(col, {})["dtype"] = str(df.dtypes.ix[col])
        return d
    return wrap_request(request, response, resp(name))


@route('/compilemodel')
def compilemodel():
    def resp(req, modelname, mode):
        print "Request: %s\n" % req
        req = simplejson.loads(req)
        returnobj = modelcompile.gen_model(req, modelname, mode)
        print returnobj[1]
        return returnobj[1]
    modelname = request.query.get('modelname', 'autorun')
    mode = request.query.get('mode', 'estimate')
    req = request.query.get('json', '')
    return wrap_request(request, response, resp(req, modelname, mode))


@route('/execmodel')
def execmodel():
    return tasks.execmodel.delay(request.query).id


@route('/execmodel/<task_id>')
@catch_exceptions
def get_result(task_id):
    res = tasks.app.AsyncResult(task_id)
    ready = res.ready()
    if ready:
        data = res.get()
    else:
        data = res._get_task_meta()["result"]
    return {"ready": ready, "data": data}


# thinking it's easier to have different tasks for models and sims
@route('/run_sim')
def run_sim():
    pass
    return tasks.run_sim.delay(request.query.json).id


@route('/simulationids')
def simulationids():
    flowerAPI = "http://localhost:5555/api/"
    h = httplib2.Http(".cache")
    (resp_headers, content) = h.request(flowerAPI + """
        tasks?limit=100&worker=All&type=urbansim.server.tasks.execmodel
        &state=All""", "GET")
    content = json.loads(content)
    print content.keys()
    return {'ids': content.keys()}


def pandas_statement(table, where, sort, orderdesc, groupby, metric,
                     limit, page):
    if where:
        where = "[DSET.fetch('%s').apply(lambda x: bool(%s),axis=1)]" % (
            table, where)
    else:
        where = ""
    if sort and orderdesc:
        sort = ".sort('%s',ascending=False)" % sort
    if sort and not orderdesc:
        sort = ".sort('%s',ascending=True)" % sort
    if not sort and orderdesc:
        sort = ".sort_index(ascending=False)"
    if not sort and not orderdesc:
        sort = ".sort_index(ascending=True)"
    if limit and page:
        # limit = ".iloc[%s*(%s-1):%s*%s]" % (limit,page,limit,page)
        limit = ".head(%s*%s).tail(%s)" % (limit, page, limit)
    elif limit:
        limit = ".head(%s)" % limit
    else:
        limit = ""
    s = "DSET.fetch('%s')%s" % (table, where)
    if groupby and metric:
        s = s + ".groupby('%s').%s.reset_index()" % (groupby, metric)
    s = s + "%s%s" % (sort, limit)

    print "Executing %s\n" % s
    recs = eval(s)
    return recs


@route('/datasets/<name>')
def datasets_records(name):
    def resp(name):
        limit = int(request.query.get('limit', 10))
        order_by = request.query.get('order_by', None)
        where = request.query.get('query', None)
        ascending = True
        if order_by and order_by[0] == "-":
            order_by = order_by[1:]
            ascending = False
        groupby = request.query.get('groupby', None)
        metric = request.query.get('metric', None)
        page = request.query.get('page', None)
        recs = pandas_statement(
            name, where, order_by, not ascending, groupby, metric, limit, page)
        d = {"recs": [recs.ix[i].values.tolist() for i in recs.index]}
        d["labels"] = recs.columns.tolist()
        return d
    return wrap_request(request, response, resp(name))


def start_service(dset, port=8765, host='localhost'):
    global DSET
    DSET = dset
    run(host=host, port=port, debug=True, server='tornado')

if __name__ == '__main__':
    start_service()

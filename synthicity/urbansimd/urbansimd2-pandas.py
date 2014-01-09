from bottle import Bottle, route, run, response, hook, request, post
import string, json, cPickle, sys, math, time, decimal
from decimal import Decimal
from django.http import HttpResponse
from django.conf import settings
import pandas as pd, numpy
import dataset
import simplejson

def jsonp(request, dictionary):
  if (request.query.callback):
    return "%s(%s)" % (request.query.callback, dictionary)
  return dictionary

def encode_float32(obj):
  if isinstance(obj, numpy.float32):
    return float(obj)
  if isinstance(obj, numpy.int32):
    return int(obj)
  if isinstance(obj, numpy.bool_):
    return bool(obj)
  print type(obj)
  raise TypeError(repr(obj) + " is not JSON serializable")

def wrap_request(request,response,obj):
  if (request.query.callback): response.content_type = "application/javascript"
  s = simplejson.dumps(obj,ignore_nan=True,default=encode_float32)
  print "Response: %s\n" % s
  return jsonp(request,s)


@route('/datasets')
def list_datasets():
  def resp():
    return [x[1:] for x in DSET.store.keys()]
  return wrap_request(request,response,resp())

@route('/datasets/<name>/info')
def datasets_info(name):
  def resp(name):
    df = DSET.fetch(name)
    return {"schema": [{"label":x,"simpletype":str(df.dtypes.ix[x]),"cardinality":int(df[x].count())} \
                       for x in DSET.fetch(name).columns]}
  return wrap_request(request,response,resp(name))

@route('/datasets/<name>/summary')
def datasets_summary(name):
  def resp(name):
    df = DSET.fetch(name).describe().transpose()
    d = dict([(k, {"summary": dict([(i,float(v.ix[i])) for i in v.index])}) for k, v in df.iterrows()])
    df = DSET.fetch(name)
    for col in df.columns: d.setdefault(col,{})["dtype"] = str(df.dtypes.ix[col])
    print d
    return d
  return wrap_request(request,response,resp(name))
  
def pandas_statement(table,where,sort,orderdesc,groupby,metric,limit):
  if where: where = "[DSET.fetch('%s').apply(lambda x: bool(%s),axis=1)]" % (table,where)
  else: where = ""
  if sort and orderdesc: sort = ".sort('%s',ascending=False)" % sort
  if sort and not orderdesc: sort = ".sort('%s',ascending=True)" % sort
  if not sort and orderdesc: sort = ".sort_index(ascending=False)"
  if not sort and not orderdesc: sort = ".sort_index(ascending=True)"
  if limit: limit = ".head(%s)" % limit
  else: limit = ""
  s = "DSET.fetch('%s')%s" % (table,where)
  if groupby and metric: s = s +".groupby('%s').%s.reset_index()" % (groupby,metric)
  s = s +"%s%s" % (sort,limit)

  print "Executing %s\n" % s
  recs = eval(s)
  return recs

@route('/datasets/<name>')
def datasets_records(name):
  def resp(name):
    limit = int(request.query.get('limit',10))
    order_by = request.query.get('order_by',None)
    where = request.query.get('query',None)
    ascending = True
    if order_by and order_by[0] == "-":
      order_by = order_by[1:]
      ascending = False
    groupby = request.query.get('groupby',None)
    metric = request.query.get('metric',None)
    print groupby, metric
    recs = pandas_statement(name,where,order_by,not ascending,groupby,metric,limit)
    print recs
    d = {"recs": [recs.ix[i].values.tolist() for i in recs.index]}
    d["labels"] = recs.columns.tolist()
    print d
    return d
  return wrap_request(request,response,resp(name))

@route('/query', method=['OPTIONS','GET','POST'])
def query():
  req = request.query.json
  if (request.query.callback): response.content_type = "application/javascript"
  print "Request: %s\n" % request.query.json
  req = simplejson.loads(req)

  table = req['table']
  metric = req['metric']
  groupby = req['groupby']
  sort = req['sort']
  limit = req['limit']
  where = req['filter']
  orderdesc = req['orderdesc']
  jointoparcels = req['jointoparcels'] 

  if where: where = "[DSET.fetch('%s').apply(lambda x: bool(%s),axis=1)]" % (table,where)
  else: where = ""
  if sort and orderdesc: sort = ".order(ascending=False)"
  if sort and not orderdesc: sort = ".order(ascending=True)"
  if not sort and orderdesc: sort = ".sort_index(ascending=False)"
  if not sort and not orderdesc: sort = ".sort_index(ascending=True)"
  if limit: limit = ".head(%s)" % limit
  else: limit = ""
  s = "DSET.fetch('%s')%s" % (table,where)
  s = s +".groupby('%s').%s%s%s" % (groupby,metric,sort,limit)

  print "Executing %s\n" % s
  recs = eval(s)
  recs = [[int(x),float(recs.ix[x])] for x in recs.index]
  s = simplejson.dumps({'records': recs},use_decimal=True)
  print "Response: %s\n" % s
  return jsonp(request,s)

def start_service(port=8765,host='localhost'):

    try: settings.configure()
    except: pass
    run(host=host, port=port, debug=True, server='tornado')

if __name__ == '__main__':  
    global DSET
    args = sys.argv[1:]
    DSET = dataset.ParisWebDataset(args[0])
    start_service()

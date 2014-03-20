from bottle import Bottle, route, run, response, hook, request, post
import string, json, cPickle, os, sys, math, time, decimal
from decimal import Decimal
from django.http import HttpResponse
from django.conf import settings
import pandas as pd, numpy
import simplejson
sys.path.insert(0,".")
import dataset


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

def wrap_request(request,response,obj):
  if (request.query.callback): response.content_type = "application/javascript"
  s = simplejson.dumps(obj,ignore_nan=True,default=encode_float32)
  print "Response: %s\n" % s
  return jsonp(request,s)

@hook('after_request')
def enable_cors():
    """
    You need to add some headers to each request.
    Don't use the wildcard '*' for Access-Control-Allow-Origin in production.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

@route('/configs')
def list_configs():
  def resp():
    files = os.listdir(misc.configs_dir())
    return files
  return wrap_request(request,response,resp())

@route('/config/<configname>', method="GET")
def read_config(configname):
  def resp():
    c = open(os.path.join(misc.configs_dir(),configname)).read()
    return simplejson.loads(c)
  return wrap_request(request,response,resp())

@route('/config/<configname>', method="OPTIONS")
def ans_opt(configname):
    return {}

@route('/config/<configname>', method="PUT")
def write_config(configname):
  json = request.json
  def resp():
    s = simplejson.dumps(json,indent=4)
    print s
    return open(os.path.join(misc.configs_dir(),configname),"w").write(s)
  return wrap_request(request,response,resp())

@route('/charts')
def list_charts():
  def resp():
    files = os.listdir(misc.charts_dir())
    return files
  return wrap_request(request,response,resp())
    
@route('/chart/<chartname>', method="GET")
def read_config(chartname):
  def resp():
    c = open(os.path.join(misc.charts_dir(),chartname)).read()
    return simplejson.loads(c)
  return wrap_request(request,response,resp())

@route('/chart/<chartname>', method="OPTIONS")
def ans_opt(chartname):
    return {}

@route('/chart/<chartname>', method="PUT")
def write_config(chartname):
  json = request.json
  def resp():
    s = simplejson.dumps(json,indent=4)
    print s
    return open(os.path.join(misc.charts_dir(),chartname),"w").write(s)
  return wrap_request(request,response,resp())

@route('/datasets')
def list_datasets():
  def resp():
    return DSET.list_tbls()
  return wrap_request(request,response,resp())

@route('/datasets/<name>/columns')
def columns_dataset(name):
  def resp():
    return DSET.fetch(name).columns.tolist()
  return wrap_request(request,response,resp())

@route('/dataset_read')
def dataset_read():
  def resp():
    url = request.query.get('url',None)
    storename = request.query.get('outname',None)
    DSET.save_tmptbl(storename,pd.read_csv(url))
    return DSET.list_tbls()
  return wrap_request(request,response,resp())

@route('/merge_datasets/<leftname>/<rightname>/<lefton>/<righton>/<how>/<outname>')
def merge_datasets(leftname,rightname,lefton,righton,how,outname):
  def resp(leftname,rightname,lefton,righton,how,outname):
    left_index = right_index = False
    if lefton == "index":
      lefton = None
      left_index = True
    if righton == "index":
      righton = None
      right_index = True
    left = DSET.fetch(leftname)
    right = DSET.fetch(rightname)
    df = pd.merge(left,right,left_on=lefton,right_on=righton,left_index=left_index,right_index=right_index,how=how)
    DSET.save_tmptbl(outname,df)
    return DSET.list_tbls()

  return wrap_request(request,response,resp(leftname,rightname,lefton,righton,how,outname))

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
    return d
  return wrap_request(request,response,resp(name))

from synthicity.utils import misc
@route('/compilemodel')
def execmodel():
  def resp():
    print request
    print "Request: %s\n" % request.query.config
    req = simplejson.loads(request.query.config)
    returnobj = misc.gen_model(req)
    print returnobj[1]
    return returnobj[1]
  return wrap_request(request,response,resp())

from synthicity.utils import misc
@route('/execmodel')
def execmodel():
  def resp(estimate,simulate):
    print "Request: %s\n" % request.query.json
    req = simplejson.loads(request.query.json)
    if estimate: mode = "estimate"
    elif simulate: mode = "simulate"
    else: mode = "run"
    returnobj = misc.run_model(req,DSET,mode=mode)
    return returnobj
  estimate = int(request.query.get('estimate',1))
  simulate = int(request.query.get('simulate',0))
  return wrap_request(request,response,resp(estimate,simulate))
 
def pandas_statement(table,where,sort,orderdesc,groupby,metric,limit,page):
  if where: where = "[DSET.fetch('%s').apply(lambda x: bool(%s),axis=1)]" % (table,where)
  else: where = ""
  if sort and orderdesc: sort = ".sort('%s',ascending=False)" % sort
  if sort and not orderdesc: sort = ".sort('%s',ascending=True)" % sort
  if not sort and orderdesc: sort = ".sort_index(ascending=False)"
  if not sort and not orderdesc: sort = ".sort_index(ascending=True)"
  if limit and page:
    limit = ".iloc[%s*(%s-1):%s*%s]" % (limit,page,limit,page)
    limit = ".head(%s*%s).tail(%s)" % (limit,page,limit)
  elif limit: limit = ".head(%s)" % limit
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
    page = request.query.get('page',None)
    recs = pandas_statement(name,where,order_by,not ascending,groupby,metric,limit,page)
    d = {"recs": [recs.ix[i].values.tolist() for i in recs.index]}
    d["labels"] = recs.columns.tolist()
    return d
  return wrap_request(request,response,resp(name))

@route('/makechart', method=['OPTIONS','GET','POST'])
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
  
  #recs = pandas_statement(table,where,sort,orderdesc,groupby,metric,limit,"") #page argument not defined and jointoparcels not used

  if where:
      where = "[DSET.fetch('%s').apply(lambda x: bool(%s),axis=1)]" % (table, where)
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
  s = "DSET.fetch('%s')%s" % (table, where)
  s = s + ".groupby('%s').%s%s%s" % (groupby, metric, sort, limit)
  
  print "Executing %s\n" % s
  recs = eval(s)
  
  if 'key_dictionary' in req:
    key_dictionary = req['key_dictionary']
    dictionary_file = open("configs/" + key_dictionary).read()                  #not sure configs is the proper place to save dicts
    dictionary = json.loads(dictionary_file)          
    print dictionary
    recs = [[dictionary[str(int(x))],float(recs.ix[x])/1000] for x in recs.index]    #the dictionary has keys from 0 to 15, ids come from 0 to 16
  else:
    recs = [[x,float(recs.ix[x])/1000] for x in recs.index]
  
  s = simplejson.dumps([{'key':'usedforwhat', 'values': recs}], use_decimal=True)
  
  print "Response: %s\n" % s
  return jsonp(request,s)

def start_service(port=8765,host='localhost'):

    try: settings.configure()
    except: pass
    run(host=host, port=port, debug=True, server='tornado')

if __name__ == '__main__':
    global DSET
    args = sys.argv[1:]
    DSET = dataset.BayAreaDataset(args[0])
    start_service()

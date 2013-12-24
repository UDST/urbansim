from bottle import Bottle, route, run, response, hook, request, post
import string, json, cPickle, sys, math, time
from django.utils import simplejson
from decimal import Decimal
from django.http import HttpResponse
from django.conf import settings
import psycopg2
import pandas as pd
import dataset

def jsonp(request, dictionary):
  if (request.query.callback):
    return "%s(%s)" % (request.query.callback, dictionary)
  return dictionary

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

def start_service(port=8764,host='paris.urbansim.org'):

    try: settings.configure()
    except: pass
    run(host=host, port=port, debug=True, server='tornado')

if __name__ == '__main__':  
    global DSET
    args = sys.argv[1:]
    DSET = dataset.ParisWebDataset(args[0])
    start_service()

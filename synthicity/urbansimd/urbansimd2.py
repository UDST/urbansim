from bottle import Bottle, route, run, response, hook, request, post
import string, json, cPickle, sys, math, time
from django.utils import simplejson
from decimal import Decimal
from django.http import HttpResponse
from django.conf import settings
import psycopg2
   
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
  
  s = "SELECT %s, %s FROM %s t1" % (groupby,metric,table)
  if jointoparcels: s += ", parcels2010_withgeography t2"
  if where: s += " WHERE %s" % where
  if jointoparcels and where: s += " and t1.parcel_id = t2.parcel_id" 
  if jointoparcels and not where: s += " WHERE t1.parcel_id = t2.parcel_id" 
  if groupby: s += " GROUP BY %s" % "1"
  if sort: s += " ORDER BY 2"
  else: s += " ORDER BY 1"
  if orderdesc: s += " DESC"
  if limit: s += " LIMIT %s" % limit

  print "Executing %s\n" % s
  global conn
  cur = conn.cursor()
  try: cur.execute(s+";")
  except Exception as e:
    conn.rollback()
    return jsonp(request,{'error': str(e)})
  recs = cur.fetchall()
  s = simplejson.dumps({'records': recs},use_decimal=True)

  print "Response: %s\n" % s
  return jsonp(request,s)

def start_service(username,password,port=8765,host='paris.urbansim.org'):

    try: settings.configure()
    except: pass
    global conn
    conn = psycopg2.connect("dbname=bayarea user=%s password=%s port=5432 host=paris.urbansim.org"%(username,password))
    run(host=host, port=port, debug=True, server='tornado')

if __name__ == '__main__':  

    args = sys.argv[1:]
    username, password = args[0], args[1]
    start_service(username,password)

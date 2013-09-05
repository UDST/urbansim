from bottle import route, run, response, hook, request, post
import pandas as pd, numpy as np, os
import json, cPickle, sys, math, time
from django.utils import simplejson
from django.http import HttpResponse
from django.conf import settings
    
INIT = False
VARCACHE = {}

@route('/compute_variable/<table>/<variable>/<xmin>/<ymin>/<xmax>/<ymax>/<srid>')
def compute_variable(table,variable,xmin,ymin,xmax,ymax,srid=None,minmaxonly=0,sampleprop=None,qcut=0,bins=0):

    assert table in TABLEDICT

    tablename = table
    table = TABLEDICT[table]
    subset = table
    
    global VARCACHE
    print tablename, variable
    key = (tablename,variable)
    if key not in VARCACHE: 
      ldict = locals()
      exec("var=%s"%variable,globals(),ldict) 
      VARCACHE[key] = ldict['var'] # cache variable
    vector = VARCACHE[key]

    global GEOMXYS
    if not minmaxonly:
      xmin,ymin,xmax,ymax = float(xmin),float(ymin),float(xmax),float(ymax)

      a = GEOMXYS[(GEOMXYS.x >= xmin) * (GEOMXYS.x < xmax) * \
                                    (GEOMXYS.y >= ymin) * (GEOMXYS.y < ymax)].index.values
      
      print "Found %d parcels" % a.size
      vector = vector.ix[a]

    if vector.values.ndim == 2: stacked = vector.stack()
    else: stacked = vector
    
    mn = stacked.min()
    mx = stacked.max()+1
    print mn, mx

    if bins > 0 and not qcut: bins = [(mx-mn)/bins*(i+1)+mn for i in range(bins-1)]
    if bins > 0 and qcut: bins = [stacked.quantile((i+1.0)/bins) for i in range(bins-1)]

    to_json = {
      'geometry':'parcels',
      'years': None,
      'values': None,
      'min': mn,
      'max': mx,
      'bins': bins,
    }
    if not minmaxonly:
      if vector.values.ndim == 2:
          to_json['values'] = dict(zip([int(x) for x in vector.index],[list(x) for x in vector.values]))
      else:
          to_json['values'] = dict(zip([int(x) for x in vector.index],[[float(x)] for x in vector.values]))
      for k,v in to_json['values'].items():
        for x in v:
          if math.isnan(x): 
            del to_json['values'][k]
            break

    return HttpResponse(simplejson.dumps(to_json), mimetype='application/json') 


@post('/')
def serve_ajax():
    req = request.json
    variable = req['variable']
    tables = req['tables']
    bbox = req.get('bbox',[-1,-1,-1,-1])
    srid = req.get('srid',None)
    minmaxonly = req.get('minmaxonly',0)
    sampleprop = req.get('sampleprop',None)
    qcut = req.get('quantile',0)
    bins = req.get('bins',0)
    return compute_variable(tables[0],variable,bbox[0],bbox[1],bbox[2],bbox[3],srid,minmaxonly,sampleprop,qcut,bins)

def start_service(geomxys,tabledict,port,host='localhost',srid=3740,debug=1,thread=0):

    try: settings.configure()
    except: pass

    global TABLEDICT, INIT, GEOMXYS
    GEOMXYS = geomxys
    TABLEDICT = tabledict
    if not INIT:
      INIT = True
      if not thread: run(host=host, port=port, debug=True, server='tornado')
      else:
        import threading
        t = threading.Thread(target=run,kwargs={'host':host,'port':port,'server':'tornado'})
        t.daemon = True
        t.start()

if __name__ == '__main__':  

    args = sys.argv[1:]
    port = 8765

    from geopandas import *
    print time.ctime()
    df = GeoDataFrame.from_file(args[0])
    print df
    df.set_index(args[1],inplace=True)
    print time.ctime()
    centroid = df.geometry.centroid.to_crs(epsg=3857)
    print time.ctime()
    geomxys = pd.DataFrame({'x':centroid.apply(lambda x: x.x),'y':centroid.apply(lambda y: y.y)},index=centroid.index)
    print geomxys
    start_service(geomxys,{'interactive':df},port)

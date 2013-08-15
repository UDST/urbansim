from bottle import route, run, response, hook, request, post
from sqlite3 import dbapi2 as db
import pandas as pd, numpy as np, os
import json, cPickle
import numpy, string, time
from django.utils import simplejson
from django.http import HttpResponse
from django.conf import settings
from bayarea import dataset
import random
import sys, math
from utils import misc
    
AVAILABLEYEARS = [2010]
DEBUG = True
INIT = False

@route('/compute_variable/<table>/<variable>/<xmin>/<ymin>/<xmax>/<ymax>/<srid>')
def compute_variable(table,variable,xmin,ymin,xmax,ymax,srid=None,minmaxonly=0,sampleprop=None,qcut=0,bins=0):
    global DEBUG
    if DEBUG: print table, variable, xmin, ymin, xmax, ymax

    assert table in TABLEDICT

    table = TABLEDICT[table]

    if not minmaxonly:
      xmin,ymin,xmax,ymax = float(xmin),float(ymin),float(xmax),float(ymax)
      srid = int(srid)

      if srid and srid <> TGTSRID:
        from utils import geomisc
        xmin,ymin = geomisc.coord_convert(xmin,ymin,srid,tgtsrid=TGTSRID)
        xmax,ymax = geomisc.coord_convert(xmax,ymax,srid,tgtsrid=TGTSRID)

      if SQLITE:
        sql = """select id from parcels, idx_parcels_geom idx where idx.ymax>=%f-50.0 and idx.ymin<=%f+50.0 and 
                   idx.xmax>=%f-50.0 and idx.xmin<=%f+50.0 and parcels.id = idx.pkid""" % (ymin,ymax,xmin,xmax)
        if DEBUG: print sql
        res = CURSOR.execute(sql)
        a = np.array([x[0] for x in res],dtype="int32")
      
      else:
        if DEBUG: print srid, xmin, xmax, ymin, ymax
        a = PARCELXYS[(PARCELXYS.x >= xmin) * (PARCELXYS.x < xmax) * \
                                    (PARCELXYS.y >= ymin) * (PARCELXYS.y < ymax)].index.values
      
      if DEBUG: print "Found %d parcels" % a.size
      subset = table.ix[a]

    else:
      subset = table
      if sampleprop and len(table.index) > 600000:
        sample = random.sample(subset.index,int(len(subset.index)*sampleprop))
        subset = subset.ix[sample] 

    subset = subset.dropna(how='all')
    variable = string.replace(variable,'_YEARS_','[%s]' % AVAILABLEYEARSSTR)
    if DEBUG: print variable

    ldict = locals()
    exec("var=%s"%variable,globals(),ldict) 
    vector = ldict['var']

    if vector.values.ndim == 2: stacked = vector.stack()
    else: stacked = vector
    
    mn = stacked.min()
    mx = stacked.max()+1
    if DEBUG: print mn, mx

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
    if vector.values.ndim == 2:
      to_json['years'] = AVAILABLEYEARS
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
      if DEBUG: print to_json['values'].items()[:10]

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

def start_service(sqlitefile,tabledict,port,host='localhost',srid=3740,sqlite=0,debug=1,thread=0):

    global DEBUG
    DEBUG = debug

    try: settings.configure()
    except: pass

    if sqlite:
      conn = db.connect(sqlitefile)
      cur = conn.cursor()

    global AVAILABLEYEARSSTR
    AVAILABLEYEARSSTR = string.join([str(x) for x in AVAILABLEYEARS],sep=',')

    global CURSOR, TABLEDICT, TGTSRID, SQLITE, PARCELXYS, INIT
    if sqlite: CURSOR = cur
    else: PARCELXYS = sqlitefile
    TABLEDICT = tabledict
    TGTSRID = srid
    SQLITE = sqlite
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
    inputfile, sqlitefile, port = args    

    print "Reading baseyear data"
    dset = dataset.Dataset(inputfile)

    print "Creating buildings table"
    buildings = pd.merge(dset.fetch('parcels',direct=1),dset.fetch('buildings',direct=1),
                                           left_index=True,right_on='parcel_id')
    buildings = buildings.set_index('parcel_id',drop=False)
    print "Creating parcels table"
    parcels = dset.fetch('parcels')
    tabledict = {'buildings':buildings,'parcels':parcels}

    start_service(sqlitefile,tabledict,port)

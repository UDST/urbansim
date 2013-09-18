from django.contrib.gis.gdal import SpatialReference, CoordTransform
from django.contrib.gis.geos import Point
import numpy as np
from multiprocessing import Pool

def coord_convert(x,y,srcsrid=4326,tgtsrid=3740):
    gcoord = SpatialReference(srcsrid)
    mycoord = SpatialReference(tgtsrid)
    trans = CoordTransform(gcoord, mycoord)
    pt = Point(float(x), float(y), srid=srcsrid)
    try: pt.transform(trans)
    except: return (np.nan,np.nan)
    return (pt.x,pt.y)

def np_coord_convert(a):
    return np.array(coord_convert(a[0],a[1]))
   
def np_coord_convert_all(a):
    # old version is single threaded
    #return np.apply_along_axis(np_coord_convert,axis=1,arr=a).astype('float32')
    pool = Pool(processes=8)
    a = np.array(pool.map(np_coord_convert,a),dtype="float32") 
    return a

	#bats['origpoint'] = bats['origpoint'].apply(lambda x: x.decode('hex')).apply(shapely.wkb.loads)
    #bats['destpoint'] = bats['destpoint'].apply(lambda x: x.decode('hex')).apply(shapely.wkb.loads)

def convert_df(df,xname='x',yname='y',srcsrid=3740,tgtsrid=4326):
    x = df[xname].values
    y = df[yname].values
    for i in range(len(df.index)):
      x[i], y[i] = coord_convert(x[i],y[i],srcsrid=srcsrid,tgtsrid=tgtsrid)
    df[xname] = x
    df[yname] = y
    return df

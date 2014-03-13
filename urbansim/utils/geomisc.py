import random
from multiprocessing import Pool

import numpy as np
import shapely

from django.contrib.gis.gdal import SpatialReference, CoordTransform
from django.contrib.gis.geos import Point


def coord_convert(x, y, srcsrid=4326, tgtsrid=3740):
    gcoord = SpatialReference(srcsrid)
    mycoord = SpatialReference(tgtsrid)
    trans = CoordTransform(gcoord, mycoord)
    pt = Point(float(x), float(y), srid=srcsrid)
    try:
        pt.transform(trans)
    except:
        return (np.nan, np.nan)
    return (pt.x, pt.y)


def np_coord_convert(a):
    return np.array(coord_convert(a[0], a[1]))


def np_coord_convert_all(a):
    # old version is single threaded
    # return
    # np.apply_along_axis(np_coord_convert,axis=1,arr=a).astype('float32')
    pool = Pool(processes=8)
    a = np.array(pool.map(np_coord_convert, a), dtype="float32")
    return a

        #bats['origpoint'] = bats['origpoint'].apply(lambda x: x.decode('hex')).apply(shapely.wkb.loads)
    #bats['destpoint'] = bats['destpoint'].apply(lambda x: x.decode('hex')).apply(shapely.wkb.loads)


def convert_df(df, xname='x', yname='y', srcsrid=3740, tgtsrid=4326):
    x = df[xname].values
    y = df[yname].values
    for i in range(len(df.index)):
        x[i], y[i] = coord_convert(
            x[i], y[i], srcsrid=srcsrid, tgtsrid=tgtsrid)
    df[xname] = x
    df[yname] = y
    return df

INVALID_X = -9999
INVALID_Y = -9999


def get_random_point_in_polygon(poly):
    (minx, miny, maxx, maxy) = poly.bounds
    p = shapely.geometry.Point(INVALID_X, INVALID_Y)
    while not poly.contains(p):
        p_x = random.uniform(minx, maxx)
        p_y = random.uniform(miny, maxy)
        p = shapely.geometry.Point(p_x, p_y)
    return p

import scipy.spatial as ss


class NN:

    def __init__(self, x, y):
        self.kd = ss.cKDTree(np.array(zip(x, y)))

    def query(self, x, y):
        return self.kd.query(zip(x, y))


def spatial_join_nearest(df1, x1, y1, df2, x2, y2):
    df1 = df1.dropna(subset=[x1, y1])
    df2 = df2.dropna(subset=[x2, y2])
    nn = NN(df1[x1], df1[y1])
    ret = nn.query(df2[x2], df2[y2])
    indexes = ret[1]
    print "Maximum distance = %.3f" % np.amax(ret[0])
    return df1.index.values[indexes], ret[0]

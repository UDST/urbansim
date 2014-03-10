import os

import pandas as pd

from utils import misc


def fetch_data(sql, dbname, outname):
    host, user = 'localhost', 'urbanvision'
    cmd = """psql -h %s -U %s -w -c "copy (%s) to stdout with csv header" %s > %s""" % \
          (host, user, sql, dbname, outname)
    os.system(cmd)

store = pd.HDFStore(os.path.join(misc.data_dir(), 'baseyeardata.h5'))

for sql, dbname, outname, hdf5name, keyname in [
    ("""select *, ST_X(ST_TRANSFORM(SETSRID(ST_POINT(longitude,latitude),4326),3740)) as x,
                  ST_Y(ST_TRANSFORM(SETSRID(ST_POINT(longitude,latitude),4326),3740)) as y
                                       from nets2011_digestformodel""",'california','nets.csv','nets','dunsnumber'),
    ('select * from zones_pemsbuffers','bayarea','zones_buffers.csv','zones_buffers','zone_id'),
    ('select *, st_astext(the_geom) as txt_geom from zones','bayarea','zones.csv','zones','zone_id'),
    ("""select *,st_x(centroid) as x,st_y(centroid) as y,st_astext(st_simplify(the_geom,2)) as txt_geom
                                 from parcels2010_withgeography""",
                                 'bayarea','parcels.csv','parcels','parcel_id'),
    ("select * from zoning_for_parcels(1,'2010-01-01 00:00:00')",
                                 'bayarea','zoning_for_parcels.csv','zoning_for_parcels','parcel'),
    ("select * from zoning_join",'bayarea','zoning.csv','zoning','id'),
    ('select * from apts_large','bayarea','apartments.csv','apartments',None),
    ('select * from households','bayarea','households.csv','households','household_id'),
    #('select n.*, coalesce(api11,0) as api11, coalesce(api10,0) as api10, coalesce(\\"Violent crime\\",-1) as violent, coalesce(\\"Property crime\\",-1) as property, county.name as county from pemsnodes n join node_geography ng on (n.id=ng.node_id) left join cityandcrime cc on (ng.city_id = cc.gid) left join county on (ng.county_id = county.gid) left join schools on (schools.gid = school_id)','sandbox','pemsnodes.csv','nodes','id'),
    ('select * from costar c','sandbox','costar.csv','costar','costar_id'),
    ('select * from buildings2010_base','bayarea','buildings.csv','buildings','building_id'),
    ('select * from home_sales2008_2011','bayarea','homesales.csv','homesales','RecordID'),
    ('select * from batshh','sandbox','bats_hhfile.csv','batshh','hhid'),
    ('select * from bats','sandbox','bats_trips.csv','bats','bats_id'),
]:
    print "Fetching data for %s" % hdf5name
    outname = os.path.join(misc.data_dir(), outname)
    fetch_data(sql, dbname, outname)
    store[hdf5name] = pd.read_csv(os.path.join(misc.data_dir(), outname),
                                  index_col=keyname)

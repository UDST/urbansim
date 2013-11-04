import pandas as pd, numpy as np, os, sys, string
from utils import misc

instore = pd.HDFStore(os.path.join(misc.data_dir(),'baseyeardata.h5'))
outstore = pd.HDFStore(os.path.join(misc.data_dir(),'baseyeardata_condensed.h5'))

for tblname in instore.keys():
    tblname = string.replace(tblname,'/','')
    if tblname in ['nodes']: continue
    print "\n\nCondensing: " + tblname
    tbl = instore[tblname]
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in instore[tblname].columns:
        if not os.popen('find | grep "\.json$" | grep -v scripts | xargs grep %s' % colname).read():
            del tbl[colname]
        elif colname in ['geom','txt_geom','nodeid']:
            del tbl[colname]
        else: 
            if tbl[colname].dtype == np.float64: newtbl[colname] = tbl[colname].astype('float32') 
            elif tbl[colname].dtype == np.int64: newtbl[colname] = tbl[colname].astype('int32')
            else: newtbl[colname] = tbl[colname]
    print newtbl.columns
    print newtbl
    outstore[tblname] = newtbl

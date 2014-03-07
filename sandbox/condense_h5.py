import pandas as pd, numpy as np, os, sys, string
from utils import misc

# this code is platform specific!
args = sys.argv[1:]

instore = pd.HDFStore(os.path.join(misc.data_dir(),args[0]))
outstore = pd.HDFStore(os.path.join(misc.data_dir(),args[1]))

for tblname in instore.keys():
    tblname = string.replace(tblname,'/','')
    print "\n\nCondensing: " + tblname
    tbl = instore[tblname]
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in instore[tblname].columns:
        if not os.popen('find | grep "\.json$" | grep -v scripts | xargs grep %s' % colname).read():
            del tbl[colname]
        elif colname in ['geom','txt_geom']:
            del tbl[colname]
        else: 
            if tbl[colname].dtype == np.float64: newtbl[colname] = tbl[colname].astype('float32') 
            elif tbl[colname].dtype == np.int64: newtbl[colname] = tbl[colname].astype('int32')
            else: newtbl[colname] = tbl[colname]
    print newtbl.columns
    print newtbl
    outstore[tblname] = newtbl

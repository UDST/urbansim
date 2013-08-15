import pandas as pd, numpy as np, statsmodels.api as sm
from synthicity.utils import misc
from modelspec import spec, fetch_table
import time, copy

def estimate(dataset,config,year=None,show=True,variables=None):
  simulate(dataset,config,year,show,variables) 

def simulate(dataset,config,year=None,show=True,variables=None):
  _tbl_ = fetch_table(dataset,config)

  t1 = time.time()
    
  _tbl_ = spec(_tbl_,config,dset=dataset,newdf=False)
  if 'show' in config and config['show']: print _tbl_.describe()
  if "writetotmp" in config: dataset.save_tmptbl(config["writetotmp"],_tbl_)

  print "Finished executing in %f seconds" % (time.time()-t1)

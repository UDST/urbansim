import numpy as np, pandas as pd, time, sys
import cPickle, os
from synthicity.utils import misc

class Networks:

  def __init__(self,filenames,factors,maxdistances,twoway):
    if not filenames: return
    from pyaccess.pyaccess import PyAccess
    self.pya = PyAccess()
    self.pya.createGraphs(len(filenames))
    for num,filename,factor,maxdistance,twoway in zip(range(len(filenames)),filenames,factors,maxdistances,twoway):
      net = cPickle.load(open(filename))
      self.pya.createGraph(num,net['nodeids'],net['nodes'],net['edges'],net['edgeweights']*factor,twoway=twoway)
      self.nodeids = net['nodeids']
      self.pya.precomputeRange(maxdistance,num)

  def accvar(self,df,distance,node_ids=None,xname='x',yname='y',vname=None,agg="AGG_SUM",decay="DECAY_LINEAR"):
    assert self.pya # need to generate pyaccess first
    pya = self.pya
    if type(agg) == type(""): agg = getattr(pya,agg)
    if type(decay) == type(""): decay = getattr(pya,decay)
    
    if node_ids is None:
      xys = np.array(df[[xname,yname]],dtype="float32")
      node_ids = pya.XYtoNode(xys,distance=distance)

    pya.initializeAccVars(1)
    num = 0
    aggvar = df[vname].astype('float32') if vname is not None else np.ones(len(df.index),dtype='float32')
    pya.initializeAccVar(num,[node_ids],aggvar,preaggregate=0)
    nodeattr = pya.getAllAggregateAccessibilityVariables(distance,num,agg,decay)
    return pd.Series(nodeattr,index=self.nodeids)
  
  def addnodeid(self,df,tosrid=0):
      try: xys = np.array(df[['x','y']],dtype="float32")
      except: xys = np.array(df[['X','Y']],dtype="float32")
        
      if tosrid: # convert coordinate system
          from synthicity.utils import geomisc
          print "Converting srids (potentially takes a long time)"
          xys = geomisc.np_coord_convert_all(xys)
          print "Finished conversion"

      for gno in range(self.pya.numgraphs): df['_node_id%d'%gno] = pd.Series(self.pya.XYtoNode(xys,gno=gno),index=df.index)
      df['_node_id'] = pd.Series(self.pya.getGraphIDS()[df['_node_id0']],index=df.index) # assign the external id as well
      return df

NETWORKS = None
def estimate(dataset,config,year=None,show=True,variables=None):
  simulate(dataset,config,year,show,variables) 

def simulate(dset,config,year=None,show=True,variables=None):

  global NETWORKS
  if not NETWORKS:
    assert 'networks' in config
    netconfig = config['networks']
    assert 'filenames' in netconfig and 'factors' in netconfig and 'maxdistances' in netconfig and 'twoway' in netconfig
    NETWORKS = Networks([os.path.join(misc.data_dir(),x) for x in netconfig['filenames']],
                    factors=netconfig['factors'],maxdistances=netconfig['maxdistances'],twoway=netconfig['twoway'])
  
  t1 = time.time()
    
  if "ind_vars" not in config: raise Exception("No ind_vars specification")
  if "var_lib" not in config: raise Exception("All network variables are defined in local var_lib")
  _tbl_ = pd.DataFrame(index=NETWORKS.nodeids)
  for varname in config["ind_vars"]:
    expression = config["var_lib"][varname]
    _tbl_[varname] = eval(expression).astype('float')
  
  if 'show' in config and config['show']: print _tbl_.describe()
  if "writetotmp" in config: dset.save_tmptbl(config["writetotmp"],_tbl_)

  print "Finished executing in %f seconds" % (time.time()-t1)

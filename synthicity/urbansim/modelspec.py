import time, copy, json
import pandas as pd, numpy as np, statsmodels.api as sm

def fetch_table(dset,config,simulate=0):

  assert 'table' in config
  table = eval(config['table'])
  if simulate and "table_sim" in config: table = eval(config['table_sim'])

  if 'filters' in config:
    for filter in config['filters']:
      _tbl_ = table
      table = table[eval(filter)]
  
  if simulate == 0 and 'estimate_filters' in config:
    for filter in config['estimate_filters']:
      _tbl_ = table
      table = table[eval(filter)]
  
  if simulate == 1 and 'simulate_filters' in config:
    for filter in config['simulate_filters']:
      _tbl_ = table
      table = table[eval(filter)]

  return table
  
def merge(dset,table,config):  
  if 'merge' in config:
    t1 = time.time()
    d = copy.copy(config['merge'])
    right = eval(d['table'])
    del(d['table'])
    table = pd.merge(table,right,**d) 
    print "Done merging land use and choosers in %f" % (time.time()-t1)
  return table

def calcvar(table,config,dset,varname):

  t1 = time.time()
  if "var_lib_file" in config:
      var_lib = json.loads(open(config["var_lib_file"]).read())
      if type(var_lib) <> type({}): raise Exception("Variable library must be of type dictionary") 
      config["var_lib"] = dict(var_lib.items()+config.get("var_lib",{}).items())

  if "var_lib" in config:
    _tbl_ = table
    if varname in config["var_lib"]:
      expression = config["var_lib"][varname]
      ret = eval(expression).astype('float')
      #print "Computed %s in %.3fs" % (varname, time.time()-t1)
      return ret

  #print "Returning column %s" % varname
  if varname in table.columns: return table[varname]

  raise Exception("Data column not found - '%s'"%varname)

def spec(segment,config,dset=None,submodel=None,newdf=True):
  t1 = time.time()
  if submodel: submodel = str(submodel)

  if newdf: est_data = pd.DataFrame(index=segment.index)
  else: est_data = segment
  if "submodel_vars" in config and submodel in config["submodel_vars"]:
    for varname in config["submodel_vars"][submodel]:
      est_data[varname] = calcvar(segment,config,dset,varname)
  else: 
    if "ind_vars" not in config: raise Exception("No ind_vars specification")
    for varname in config["ind_vars"]:
      est_data[varname] = calcvar(segment,config,dset,varname)

  est_data = est_data.fillna(0)

  if 'add_constant' in config:
    est_data = sm.add_constant(est_data,prepend=False)

  print "Specifying model in %f" % (time.time()-t1)
  return est_data

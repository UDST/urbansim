import pandas as pd, numpy as np, statsmodels.api as sm
from synthicity.utils import misc
from modelspec import spec, fetch_table, merge
import time, copy

def estimate(dset,config,year=None,show=True,simulate=0,variables=None):

  returnobj = {}
  t1 = time.time()
  
  buildings = fetch_table(dset,config,simulate)

  buildings = merge(dset,buildings,config)

  assert "output_names" in config
  output_csv, output_title, coeff_name, output_varname = config["output_names"]

  print "Finished specifying in %f seconds" % (time.time()-t1)
  t1 = time.time()

  simrents = []
  segments = [(None,buildings)]
  if 'segment' in config: segments = buildings.groupby(config['segment'])
  
  for name, segment in segments:
    
    est_data = spec(segment,config,submodel=name,dset=dset)
    if name is not None: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
    else: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv, output_title, coeff_name

    if not simulate:

      assert "dep_var" in config
      if type(config["dep_var"]) == type(u""):
        depvar = segment[config["dep_var"]]
      else: # dependent variable already got computed and substituted by patsy
        depvar = config["dep_var"]
      if "dep_var_transform" in config: depvar = depvar.apply(eval(config['dep_var_transform']))
      
      if name: print "Estimating hedonic for %s with %d observations" % (name,len(segment.index))
      if show : print est_data.describe()
      dset.save_tmptbl("EST_DATA",est_data)

      model = sm.OLS(depvar,est_data)
      results = model.fit()
      if show: print results.summary()

      tmp_outcsv = output_csv if name is None else output_csv%name
      tmp_outtitle = output_title if name is None else output_title%name
      misc.resultstocsv((results.rsquared,results.rsquared_adj),est_data.columns,
                        zip(results.params,results.bse,results.tvalues),tmp_outcsv,hedonic=1,
                        tblname=output_title)
      d = {}
      d['rsquared'] = results.rsquared
      d['rsquared_adj'] = results.rsquared_adj
      d['columns'] =  est_data.columns.tolist()
      d['est_results'] =  zip(results.params,results.bse,results.tvalues)
      returnobj[name] = d

      dset.store_coeff(tmp_coeffname,results.params.values,results.params.index)

    else:

      print "Generating rents on %d buildings" % (est_data.shape[0])
    
      vec = dset.load_coeff(tmp_coeffname)
      vec = np.reshape(vec,(vec.size,1))
      rents = est_data.dot(vec).astype('f4')
      if "output_transform" in config: rents = rents.apply(eval(config['output_transform']))
   
      simrents.append(rents[rents.columns[0]])

  if simulate:
    simrents = pd.concat(simrents)
    dset.buildings[output_varname] = simrents.reindex(dset.buildings.index)
    dset.store_attr(output_varname,year,simrents)

  print "Finished executing in %f seconds" % (time.time()-t1)
  return returnobj

def simulate(dset,config,year,variables=None,show=False):
  return estimate(dset,config,year,show=show,simulate=1,variables=variables)

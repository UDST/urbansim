import pandas as pd, numpy as np, statsmodels.api as sm
from synthicity.utils import misc
{% from 'modelspec.py' import MERGE, SPEC, TABLE with context %}
import time, copy

def {{modelname}}_{{template_mode}}(dset,year=None,show=True):

  assert "{{model}}" == "hedonicmodel" # should match!
  returnobj = {}
  t1 = time.time()
  
  # TEMPLATE configure table
  {{ TABLE("buildings")|indent(2) }}
  # ENDTEMPLATE

  {% if merge -%}
  # TEMPLATE merge
  {{- MERGE("buildings",merge) | indent(2) }}
  # ENDTEMPLATE
  {% endif %}

  # TEMPLATE specifying output names
  output_csv, output_title, coeff_name, output_varname = {{output_names}}
  # ENDTEMPLATE

  print "Finished specifying in %f seconds" % (time.time()-t1)
  t1 = time.time()

  {% if not template_mode == "estimate" -%}
  simrents = []
  {% endif -%}
  {% if segment is not defined -%}
  segments = [(None,buildings)]
  {% else %}

  # TEMPLATE creating segments
  segments = buildings.groupby({{segment}})
  # ENDTEMPLATE
  {% endif  %}
  
  for name, segment in segments:
    
    # TEMPLATE computing vars
    {{ SPEC("segment","est_data",submodel="name") | indent(4) }}
    # ENDTEMPLATE

    if name is not None: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
    else: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv, output_title, coeff_name
    {% if template_mode == "estimate" %}
    
    # TEMPLATE dependent variable
    depvar = segment["{{dep_var}}"]
    {% if dep_var_transform is defined -%}
    depvar = depvar.apply({{dep_var_transform}})
    # ENDTEMPLATE
    {% endif %}

    if name: print "Estimating hedonic for %s with %d observations" % (name,len(segment.index))
    if show: print est_data.describe()
    dset.save_tmptbl("EST_DATA",est_data) # in case we want to explore via urbansimd

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
    {% else %} {# SIMULATE #} 
    print "Generating rents on %d buildings" % (est_data.shape[0])
    vec = dset.load_coeff(tmp_coeffname)
    vec = np.reshape(vec,(vec.size,1))
    rents = est_data.dot(vec).astype('f4')
    {% if output_transform -%}
    rents = rents.apply({{output_transform}})
    {% endif -%}
    simrents.append(rents[rents.columns[0]])
    {% endif %}
  
  {% if not template_mode == "estimate" -%}
  simrents = pd.concat(simrents)
  dset.buildings[output_varname] = simrents.reindex(dset.buildings.index)
  dset.store_attr(output_varname,year,simrents)

  {% endif -%}
  
  print "Finished executing in %f seconds" % (time.time()-t1)
  return returnobj

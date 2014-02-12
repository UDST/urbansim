import time, copy, json
import pandas as pd, numpy as np, statsmodels.api as sm
from patsy import dmatrices

{% macro TABLE(tblname) %}
{% if not template_mode == "estimate" and table_sim %}
{{tblname}} = {{table_sim}}
{% else %}
{{tblname}} = {{table}}
{% endif %}
{% if filters %}
{% for filter in filters %}
{{tblname}} = {{tblname}}[{{filter|replace("_tbl_",tblname)}}]
{% endfor %}
{% endif %}
{% if template_mode == "estimate" and estimate_filters %}
{% for filter in estimate_filters %}
{{tblname}} = {{tblname}}[{{filter|replace("_tbl_",tblname)}}]
{% endfor %}
{% endif %}
{% if not template_mode == "estimate" and simulate_filters %}
{% for filter in simulate_filters %}
{{tblname}} = {{tblname}}[{{filter|replace("_tbl_",tblname)}}]
{% endfor %}
{% endif %}
{% endmacro %}
  
{% macro MERGE(tablename,merged) %} 
t_m = time.time()
{{tablename}} = pd.merge({{tablename}},{{merged.table}},**{{merged|droptable}})
print "Finished with merge in %f" % (time.time()-t_m)
{% endmacro %}

{% macro CALCVAR(table,varname,var_lib) %}
{% if varname in var_lib %}
({{var_lib[varname]|replace("_tbl_",table)}}).astype('float')
{% else %}
{{table}}["{{varname}}"]
{% endif %}
{% endmacro %}

{% macro SPEC(inname,outname,submodel=None,newdf=True) %} 
{%- if patsy %}
  print config['patsy']
  # use patsy to specify the data
  y, X = dmatrices(config['patsy'], data=inname, return_type='dataframe')
  if 'dep_var' in config or 'dep_var_transform' in config:
    print "WARNING: using patsy, dep_var and dep_var_transform are ignored"
    config['dep_var'] = y
    if 'dep_var_transform' in config: del config['dep_var_transform']
    return X
{% endif -%}
{% if newdf %}
{{outname}} = pd.DataFrame(index={{inname}}.index)
{% else -%}
{{outname}} = {{inname}}
{% endif %}
{% if submodel_vars and submodel in submodel_vars %}
    for varname in config["submodel_vars"][submodel]:
      est_data[varname] = CALCVAR(inname,varname,var_lib)
{% else %}
{% for varname in ind_vars %}
{{outname}}["{{varname}}"] = {{CALCVAR(inname,varname,var_lib)-}}
{% endfor %}
{% endif -%}
{% if add_constant %}
{{outname}} = sm.add_constant({{outname}},prepend=False)
{% endif %}
{{outname}} = {{outname}}.fillna(0)
{% endmacro %}

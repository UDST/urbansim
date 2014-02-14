import pandas as pd, numpy as np, statsmodels.api as sm
from synthicity.utils import misc
{% from 'modelspec.py' import MERGE, SPEC, TABLE with context %}
import time, copy

def {{modelname}}_run(dset,year=None,show=True):
  assert "{{template_mode}}" == "run"

  # TEMPLATE configure table
  {{ TABLE("_tbl_")|indent(2) }}
  # ENDTEMPLATE

  t1 = time.time()
    
  # TEMPLATE computing vars
  {{ SPEC("_tbl_","_tbl_",newdf=False) | indent(2) }}
  # ENDTEMPLATE
  {% if show -%}
  print _tbl_.describe()
  {% endif -%}
  {% if writetotmp -%}
  dset.save_tmptbl("{{writetotmp}}",_tbl_)
  {% endif %}

  print "Finished executing in %f seconds" % (time.time()-t1)
  return misc.pandasdfsummarytojson(_tbl_.describe())

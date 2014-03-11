{% from 'modelspec.py' import IMPORTS, MERGE, SPEC, TABLE with context %}
{{ IMPORTS() }}
from synthicity.utils import networks

def {{modelname}}_run(dset,year=None,show=True):
  assert "{{template_mode}}" == "run"

  if not networks.NETWORKS:
    networks.NETWORKS = networks.Networks([os.path.join(misc.data_dir(),x) for x in {{networks.filenames}}],
                    factors={{networks.factors}},maxdistances={{networks.maxdistances}},twoway={{networks.twoway}},
    {% if networks.impedances %}
                impedances={{networks.impedances}})
    {% else %}
                impedances=None)
    {% endif %}

  t1 = time.time()

  {% for update_tbl in update_xys -%}
  {{update_tbl}} = dset.add_xy({{update_tbl}})
  {% endfor %}

  NETWORKS = networks.NETWORKS # make available for the variables
  _tbl_ = pd.DataFrame(index=pd.MultiIndex.from_tuples(networks.NETWORKS.nodeids,names=['_graph_id','_node_id']))
  {{ SPEC("_tbl_","_tbl_",newdf=False) | indent(2) }}

  {% if show -%}
  if show: print _tbl_.describe()
  {% endif %}
  _tbl_ = _tbl_.reset_index().set_index(networks.NETWORKS.external_nodeids)
  {% if writetotmp -%}
  dset.save_tmptbl("{{writetotmp}}",_tbl_)
  {% endif %}

  {% if writetocsv -%}
  _tbl_.to_csv(os.path.join(misc.data_dir(),"{{writetocsv}}"),index_label='node_id',float_format="%.3f")
  {% endif %}

  print "Finished executing in %f seconds" % (time.time()-t1)

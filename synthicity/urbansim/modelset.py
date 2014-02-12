import os, sys
import simplejson
from synthicity.utils import misc

{% if pathinsertcwd %}
sys.path.insert(0,".")
{% endif %}
import dataset
dset = dataset.{{dataset}}(os.path.join(misc.data_dir(),'{{datastore}}'))

{% for arg in modelstorun %}
print "Running {{arg}}"
import {{arg}}
retval = {{arg}}.{{arg}}(dset)
if retval: open(os.path.join(misc.output_dir(),"{{arg}}.json"),"w").write(simplejson.dumps(retval))
{% endfor %} 

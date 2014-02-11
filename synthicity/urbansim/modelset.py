import os, sys
from synthicity.utils import misc

{% if pathinsertcwd %}
sys.path.insert(0,".")
{% endif %}
import dataset
dset = dataset.{{dataset}}(os.path.join(misc.data_dir(),'{{datastore}}'))

{% for arg in modelstorun %}
import {{arg}}
{{arg}}.{{arg}}(dset)
{% endfor %} 

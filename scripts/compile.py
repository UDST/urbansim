import os, string, sys, json
from synthicity.utils import misc

args = sys.argv[1:]

for arg in args:
  print "Generating %s" % arg
  d = misc.gen_model(arg)
  for mode, code in d.items():
    basename = os.path.splitext(os.path.basename(arg))[0]
    open(os.path.join(misc.models_dir(),basename+'_'+mode+'.py'),'w').write(code)

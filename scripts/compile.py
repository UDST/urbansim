import os, string, sys, json
from synthicity.utils import misc

args = sys.argv[1:]

for arg in args:
  print "Generating %s" % arg
  basename, d = misc.gen_model(arg)
  for mode, code in d.items():
    open(os.path.join(misc.models_dir(),basename+'_'+mode+'.py'),'w').write(code)

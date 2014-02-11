import os, string, sys, json
from synthicity.utils import misc

args = sys.argv[1:]
assert len(args) == 1
config = json.loads(open(args[0]).read()) 

if not os.path.exists(config["modelpybase"]): os.mkdir(config["modelpybase"])
 
for arg in config['modelstorun']:
  print "Generating %s" % arg
  model, param = string.split(arg,'_')
  assert param in ["estimate","simulate"]
  estimate = param == "estimate"
  # generate
  s = misc.gen_model(config['modeljsonbase'],model,estimate)
  suffix = "estimate" if estimate else "simulate"
  open(os.path.join(config['modelpybase'],model)+'_'+suffix+'.py','w').write(s)
 
print "Generating main"
basename = os.path.splitext(os.path.basename(args[0]))[0]
s = misc.gen_model(config['modeljsonbase'],basename,False)    
open(os.path.join(config['modelpybase'],"main")+'.py','w').write(s)

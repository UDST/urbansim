import os, json, sys, time
from mrcog import dataset
from utils import misc
from urbansim import hedonicmodel, locationchoicemodel, minimodel, transitionmodel
from urbansim import locationchoice
from multiprocessing import Pool

args = sys.argv[1:]

dataset = dataset.MRCOGDataset(os.path.join(misc.data_dir(),'mrcog.h5'))
num = misc.get_run_number()

def run_model(fname,dset=dataset,show=1,estimate=1,year=2010):
  print "Running %s\n" % fname
  config = json.loads(open(fname).read()) 
  assert "model" in config
  model = eval(config["model"])
  if estimate: model.estimate(dset,config,2010,show=show)
  if config["model"] == "hedonicmodel" or not estimate: 
    t1 = time.time()
    model.simulate(dset,config,year,show=show)
    print "SIMULATED %s model in %.3f seconds" % (fname,time.time()-t1)

if __name__ == '__main__':
  print time.ctime()
  for arg in args: run_model(arg,estimate=1)
  print time.ctime()
  t1 = time.time()
  numyears = 30
  for i in range(numyears):
    t2 = time.time()
    for arg in args: run_model(arg,show=False,estimate=0,year=2010+i)
    print "Time to simulate year %d = %.3f" % (i+1,time.time()-t2)
  print "Actual time to simulate per year = %.3f" % ((time.time()-t1)/float(numyears))
  #pool = Pool(processes=len(args))
  #pool.map(run_model,args)
  dataset.save_coeffs(os.path.join(misc.runs_dir(),'run_drive_%d.h5'%num))
  dataset.save_output(os.path.join(misc.runs_dir(),'run_drive_%d.h5'%num))
  print time.ctime()

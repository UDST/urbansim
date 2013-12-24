import os, json, sys, time
from synthicity.utils import misc
import dataset

args = sys.argv[1:]

dset = dataset.MRCOGDataset(os.path.join(misc.data_dir(),'mrcog.h5'))
num = misc.get_run_number()

if __name__ == '__main__':
  print time.ctime()
  for arg in args: misc.run_model(arg,dset,estimate=1,simulate=1)
  print time.ctime()
  t1 = time.time()
  numyears = 30
  for i in range(numyears):
    t2 = time.time()
    for arg in args: misc.run_model(arg,dset,show=False,estimate=0,simulate=1,year=2010+i)
    print "Time to simulate year %d = %.3f" % (i+1,time.time()-t2)
  print "Actual time to simulate per year = %.3f" % ((time.time()-t1)/float(numyears))
  #from multiprocessing import Pool
  #pool = Pool(processes=len(args))
  #pool.map(run_model,args)
  dset.save_coeffs(os.path.join(misc.runs_dir(),'run_drive_%d.h5'%num))
  dset.save_output(os.path.join(misc.runs_dir(),'run_drive_%d.h5'%num))
  print time.ctime()

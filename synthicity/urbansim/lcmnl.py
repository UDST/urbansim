from synthicity.urbansim import interaction, mnl
import numpy as np, pandas as pd
import time

GPU = 0
EMTOL = 1e-02
MAXITER = 10000

def prep_cm_data(cmdata,numclasses):
  numobs, numvars = cmdata.shape
  newcmdata = np.zeros((numobs*numclasses,numvars*(numclasses-1)))
  for i in range(cmdata.shape[0]): 
    for j in range(1,numclasses):
      newcmdata[i*numclasses+j,(j-1)*numvars:j*numvars] = cmdata[i]
  return newcmdata

def lcmnl_estimate(cmdata,numclasses,csdata,numalts,chosen,maxiter=MAXITER,emtol=EMTOL,\
                     skipprep=False,csbeta=None,cmbeta=None):

  loglik = -999999
  if csbeta is None: csbeta = [np.random.rand(csdata.shape[1]) for i in range(numclasses)]
  if not skipprep: cmdata = prep_cm_data(cmdata,numclasses)
  if cmbeta is None: cmbeta = np.zeros(cmdata.shape[1])
  
  for i in range(maxiter):
    print "Running iteration %d" % (i+1)
    print time.ctime()

    # EXPECTATION
    print "Running class membership model"
    cmprobs = mnl.mnl_simulate(cmdata,cmbeta,numclasses,GPU=GPU,returnprobs=1)

    csprobs = []
    for cno in range(numclasses):
      tmp = mnl.mnl_simulate(csdata,csbeta[cno],numalts,GPU=GPU,returnprobs=1)
      tmp = np.sum(tmp*chosen,axis=1) # keep only chosen probs
      csprobs.append(np.reshape(tmp,(-1,1)))
    csprobs = np.concatenate(csprobs,axis=1)

    h = csprobs * cmprobs
    oldloglik = loglik
    loglik = np.sum(np.log(np.sum(h,axis=1)))
    print "current csbeta", csbeta
    print "current cmbeta", cmbeta
    print "current loglik", loglik, i+1, "\n\n"
    if abs(loglik-oldloglik) < emtol: break
    wts = h / np.reshape(np.sum(h,axis=1),(-1,1))
   
    # MAXIMIZATION

    for cno in range(numclasses):
      print "Estimating class specific model for class %d" % (cno+1)
      t1 =  time.time()
      weights=np.reshape(wts[:,cno],(-1,1))
      print weights.shape
      fit, results  = mnl.mnl_estimate(csdata,chosen,numalts,GPU=GPU,weights=weights,beta=csbeta[cno])
      print "Finished in %fs" % (time.time()-t1)
      csbeta[cno] = zip(*results)[0]
    
    print "Estimating class membership model"
    t1 =  time.time()
    fit, results = mnl.mnl_estimate(cmdata,None,numclasses,GPU=GPU,weights=wts,lcgrad=True, \
                                             beta=cmbeta,coeffrange=(-1000,1000))
    print "Finished in %fs" % (time.time()-t1)
    cmbeta = zip(*results)[0]

from synthicity.urbansim import interaction, mnl
import numpy as np, pandas as pd

GPU = 0
if GPU: interaction.enable_gpu()

EMTOL = 1e-06
MAXITER = 10000

def prep_cm_data(cmdata,numclasses):
  numobs, numvars = cmdata.shape
  newcmdata = np.zeros((numobs*numclasses,numvars*(numclasses-1)))
  for i in range(cmdata.shape[0]): 
    for j in range(1,numclasses):
      newcmdata[i*numclasses+j,(j-1)*numvars:j*numvars] = cmdata[i]
  return newcmdata

def lcmnl_estimate(cmdata,numclasses,csdata,numalts,chosen,maxiter=MAXITER,emtol=EMTOL,skipprep=False,beta=None,cmbeta=None):

  loglik = -999999
  if beta is None: beta = [np.random.rand(csdata.shape[1]) for i in range(numclasses)]
  if not skipprep: cmdata = prep_cm_data(cmdata,numclasses)
  if cmbeta is None: cmbeta = np.zeros(cmdata.shape[1])
  
  for i in range(maxiter):
    print "Running iteration %d" % (i+1)

    # EXPECTATION
    print "Running class membership model"
    print "cmbeta", cmbeta
    cmprobs = mnl.mnl_simulate(cmdata,cmbeta,numclasses,GPU=GPU,returnprobs=1)
    print "cmprobs", cmprobs

    csprobs = []
    for cno in range(numclasses):
      print "Running class specific model for class %d" % (cno+1)
      print "csbeta", beta[cno]
      print "csdata", csdata
      print numalts
      tmp = mnl.mnl_simulate(csdata,beta[cno],numalts,GPU=GPU,returnprobs=1)
      print "csprobs", tmp.shape, tmp
      tmp = np.sum(tmp*chosen,axis=1) # keep only chosen probs
      csprobs.append(np.reshape(tmp,(-1,1)))
    csprobs = np.concatenate(csprobs,axis=1)

    print csprobs
    print cmprobs
    h = csprobs * cmprobs
    print h
    oldloglik = loglik
    loglik = np.sum(np.log(np.sum(h,axis=1)))
    print "current loglik", loglik, i+1
    print "current beta", beta
    if abs(loglik-oldloglik) < emtol: break
    wts = h / np.reshape(np.sum(h,axis=1),(-1,1))
    for i in range(wts.shape[1]): print i, pd.Series(wts[:,i]).describe()
   
    # MAXIMIZATION

    for cno in range(numclasses):
      print "Estimating class specific model for class %d" % (cno+1)
      print "weights", wts[:,cno]
      fit, results  = mnl.mnl_estimate(csdata,chosen,numalts,GPU=GPU,weights=wts[:,cno],beta=beta[cno])

      beta[cno] = zip(*results)[0]
      print "done" ,beta[cno]
    
    print "Estimating class membership model"
    fit, results = mnl.mnl_estimate(cmdata,None,numclasses,GPU=GPU,weights=wts,lcgrad=True,beta=cmbeta)
    cmbeta = zip(*results)[0]

    print "beta", beta
    print "cmbeta", cmbeta

if __name__ == "__main__":

  NUMCLASSES = 2
  NUMALTS = 50
  NUMCHOICES = 5
  NUMVARS = 10 # placeholder for num variables for both CM and CS models
  csdata = np.random.random((NUMCHOICES*NUMALTS,NUMVARS))
  cmdata = np.random.random((NUMCHOICES*NUMCLASSES,NUMVARS))
  for i in range(NUMCHOICES): cmdata[i*NUMCLASSES,:] = 0
  chosen = np.zeros((NUMCHOICES,NUMALTS))
  chosen[np.arange(NUMCHOICES),(np.random.random(NUMCHOICES)*NUMALTS).astype('int')] = 1

  lcmnl_estimate(cmdata,NUMCLASSES,csdata,NUMALTS,chosen)

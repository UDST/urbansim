from synthicity.urbansim import interaction, mnl
import numpy as np, pandas as pd

GPU = 0
if GPU: interaction.enable_gpu()

EMTOL = 1e-03
MAXITER = 10000

def lcmnl_estimate(cmdata,numclasses,csdata,numalts,chosen,maxiter=MAXITER,emtol=EMTOL):

  loglik = -999999
  beta = [np.ones(csdata.shape[1]) for i in range(numclasses)]
  #cmbeta = np.ones((cmdata.shape[1]+1)*(numclasses-1)) # +1 if for asc, -1 is base alt
  cmbeta = np.ones(cmdata.shape[1])
  
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
      tmp = mnl.mnl_simulate(csdata,beta[cno],numalts,GPU=GPU,returnprobs=1)
      print "csprobs", tmp
      tmp = np.sum(tmp*chosen,axis=1) # keep only chosen probs
      csprobs.append(np.reshape(tmp,(-1,1)))
    csprobs = np.concatenate(csprobs,axis=1)

    h = csprobs * cmprobs
    oldloglik = loglik
    loglik = np.sum(np.log(np.sum(h,axis=1)))
    print "current loglik", loglik, i+1
    print "current beta", beta
    if abs(loglik-oldloglik) < emtol: break
    wts = h / np.reshape(np.sum(h,axis=1),(-1,1))
    for i in range(wts.shape[1]): print i, pd.Series(wts[:,i]).describe()
   
    # MAXIMIZATION

    print "Estimating class membership model"
    fit, results = mnl.mnl_estimate(cmdata,None,numclasses,GPU=GPU,weights=wts,lcgrad=True)
    cmbeta = zip(*results)[0]

    for cno in range(numclasses):
      print "Estimating class specific model for class %d" % (cno+1)
      fit, results  = mnl.mnl_estimate(csdata,chosen,numalts,GPU=GPU,weights=wts[:,cno])

      beta[cno] = zip(*results)[0]

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

from synthicity.urbansim import interaction, mnl
import numpy as np, pandas as pd

GPU = 0
if GPU: interaction.enable_gpu()
NUMCLASSES = 2
MAXITER = 10000
NUMALTS = 50
NUMCHOICES = 5
NUMVARS = 10 # placeholder for num variables for both CM and CS models
data = np.random.random((NUMCHOICES*NUMALTS,NUMVARS))
beta = [np.ones(NUMVARS) for i in range(NUMCLASSES)]
cmbeta = np.ones(NUMVARS)
cmdata = np.random.random((NUMCHOICES*NUMCLASSES,NUMVARS))
for i in range(NUMCHOICES): cmdata[i*NUMCLASSES,:] = 0
chosen = np.zeros((NUMCHOICES,NUMALTS))
chosen[np.arange(NUMCHOICES),(np.random.random(NUMCHOICES)*NUMALTS).astype('int')] = 1

loglik = -999999
emtol = 1e-03

for i in range(MAXITER):
  print "Running iteration %d" % (i+1)

  # EXPECTATION
  print "Running class membership model"
  print "cmbeta", cmbeta
  cmprobs = mnl.mnl_simulate(cmdata,cmbeta,NUMCLASSES,GPU=GPU,returnprobs=1)
  print "cmprobs", cmprobs

  csprobs = []
  for cno in range(NUMCLASSES):
    print "Running class specific model for class %d" % (cno+1)
    print "csbeta", beta[cno]
    tmp = mnl.mnl_simulate(data,beta[cno],NUMALTS,GPU=GPU,returnprobs=1)
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
  fit, results = mnl.mnl_estimate(cmdata,None,NUMCLASSES,GPU=GPU,weights=wts,lcgrad=True)
  cmbeta = zip(*results)[0]

  for cno in range(NUMCLASSES):
    print "Estimating class specific model for class %d" % (cno+1)
    fit, results  = mnl.mnl_estimate(data,chosen,NUMALTS,GPU=GPU,weights=wts[:,cno])
    beta[cno] = zip(*results)[0]

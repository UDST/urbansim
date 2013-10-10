from synthicity.urbansim import interaction, mnl
import numpy as np, pandas as pd

GPU = 0
NUMCLASSES = 2
MAXITER = 3
NUMALTS = 50
NUMCHOICES = 100
NUMVARS = 10 # placeholder for num variables for both CM and CS models
data = np.random.random((NUMCHOICES*NUMALTS,NUMVARS))
beta = [np.zeros(NUMVARS) for i in range(NUMCLASSES)]
cmbeta = np.zeros(NUMVARS)
cmdata = np.random.random((NUMCHOICES*NUMCLASSES,NUMVARS))
chosen = np.zeros((NUMCHOICES,NUMALTS))
chosen[np.arange(NUMCHOICES),(np.random.random(NUMCHOICES)*NUMALTS).astype('int')] = 1

for i in range(MAXITER):
  print "Running iteration %d" % (i+1)

  # EXPECTATION
  print "Running class membership model"
  print cmbeta
  cmprobs = mnl.mnl_simulate(cmdata,cmbeta,NUMCLASSES,GPU=GPU,returnprobs=1)

  csprobs = []
  for cno in range(NUMCLASSES):
    print "Running class specific model for class %d" % (cno+1)
    print beta[cno]
    tmp = mnl.mnl_simulate(data,beta[cno],NUMALTS,GPU=GPU,returnprobs=1)
    print tmp
    tmp = np.sum(tmp*chosen,axis=1) # keep only chosen probs
    csprobs.append(np.reshape(tmp,(-1,1)))
  csprobs = np.concatenate(csprobs,axis=1)

  h = csprobs * cmprobs
  loglik = np.sum(np.log(np.sum(h,axis=1)))
  print loglik
  print beta
  wts = h / np.reshape(np.sum(h,axis=1),(-1,1))
  
  # MAXIMIZATION

  print "Estimating class membership model"
  fit, results = mnl.mnl_estimate(cmdata,None,NUMCLASSES,GPU=GPU,weights=wts,lcgrad=True)
  cmbeta = zip(*results)[0]
  print cmbeta

  for cno in range(NUMCLASSES):
    print "Estimating class specific model for class %d" % (cno+1)
    print pd.Series(wts[:,cno]).describe()
    fit, results  = mnl.mnl_estimate(data,chosen,NUMALTS,GPU=GPU,weights=wts[:,cno])
    beta[cno] = zip(*results)[0]

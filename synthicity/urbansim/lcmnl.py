import pandas as pd, numpy as np
import mnl

# basic outline for estimating latent class models 

def lcmnl_estimate(memdata,specdata,chosen,numalts=None):
   print "time to estimate"
   mnl.mnl_estimate(specdata,chosen,numalts) # won't really work right now 

def lcmnl_simulate(data, coeff, numalts, GPU=0, returnprobs=0):
   # to come later
   pass

if __name__ == '__main__':
  df = pd.read_csv('dataHoldLong01.txt',sep='\t',header=None)
  mat = df.as_matrix()
  memdata = mat[:,12:19]
  specdata = mat[:,18:]
  choices = mat[:,11]
  print memdata.shape
  print specdata.shape
  print choices.shape
  print pd.Series(choices).describe()
  lcmnl_estimate(memdata,specdata,choices)

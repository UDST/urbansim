import numpy as np, pandas as pd
import time, os
from synthicity.urbansim import dataset

class ParisWebDataset(dataset.Dataset):

  def __init__(self,filename):
    super(ParisWebDataset,self).__init__(filename)
  
  def fetch(self,name,**kwargs):
    if name in self.d: return self.d[name]
    df = super(ParisWebDataset,self).fetch(name,**kwargs) 
    if "households" in name or "establishments" in name:
     
      buildingsname = "_".join(name.split('_')[:-1]+['buildings'])
      df = pd.merge(df,self.fetch(buildingsname),left_on='building_id',right_index=True)
    
    self.d[name] = df
    return df

import numpy as np, pandas as pd
import time, os
from urbansim import dataset

class MRCOGDataset(dataset.Dataset):

  def __init__(self,filename):
    super(MRCOGDataset,self).__init__(filename)
  
  def fetch_annual_job_relocation_rates(self):
    return self.store["annual_job_relocation_rates"].reset_index()

  def nhb_jobs(self): return self.jobs[self.jobs.home_based_status==0]
 
  def fetch_target_vacancies(self):
    return self.store["target_vacancies"].stack().reset_index(level=2,drop=True).unstack(level=1)
  
  def fetch_annual_employment_control_totals(self):
    return self.store["annual_employment_control_totals"].reset_index(level=2).reset_index(level=0)

  def fetch_buildings(self):
    # this is a bug in the data - need to eliminate duplicate ids - not allowed!
    buildings = self.store['buildings'].reset_index().drop_duplicates().set_index('building_id')
    buildings["dasz_id"] = self.parcels["dasz_id"][buildings.parcel_id].values
    buildings = pd.merge(buildings,self.store['building_sqft_per_job'],left_on=['dasz_id','building_type_id'],\
                                       right_index=True,how='left')
    buildings["non_residential_units"] = buildings.non_residential_sqft/buildings.building_sqft_per_job
    buildings["base_year_jobs"] = self.jobs.groupby('building_id').size()
    # things get all screwed up if you have overfull buildings
    buildings["non_residential_units"] = buildings[["non_residential_units","base_year_jobs"]].max(axis=1)
    buildings["all_units"] = buildings.residential_units + buildings.non_residential_units
    return buildings

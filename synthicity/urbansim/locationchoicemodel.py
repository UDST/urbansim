import pandas as pd, numpy as np, statsmodels.api as sm
from utils import misc
from modelspec import spec, fetch_table, calcvar, merge
import locationchoice
import os, time, copy

SAMPLE_SIZE=100

##############
#  ESTIMATION
##############

def estimate(dset,config,year,show=True,variables=None):

  choosers = fetch_table(dset,config)
  if 'est_sample_size' in config: 
    choosers = choosers.ix[np.random.choice(choosers.index, config['est_sample_size'],replace=False)]
  output_csv, output_title, coeff_name, output_varname = config["output_names"]
 
  assert 'alternatives' in config
  alternatives = eval(config['alternatives'])
  alternatives = merge(dset,alternatives,config)

  t1 = time.time()

  segments = [(None,choosers)]
  if 'segment' in config:
    for varname in config['segment']:
      if varname not in choosers.columns:
        choosers[varname] = calcvar(choosers,config,dset,varname)
    segments = choosers.groupby(config['segment'])
  for name, segment in segments:

    name = str(name)
    if name is not None: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
    else: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv, output_title, coeff_name

    assert "dep_var" in config
    depvar = config["dep_var"]
    SAMPLE_SIZE = config["alt_sample_size"] if "alt_sample_size" in config else SAMPLE_SIZE 
    sample, alternative_sample, est_params = locationchoice.mnl_interaction_dataset(
                                        segment,alternatives,SAMPLE_SIZE,chosenalts=segment[depvar])

    print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 

    data = spec(alternative_sample,config,submodel=name)
    if show: print data.describe()
    data = data.as_matrix()
    
    fnames = config['ind_vars']
    fnames = config['ind_var_names'] if 'ind_var_names' in config else fnames

    fit, results = locationchoice.estimate(data,est_params,SAMPLE_SIZE)
    
    fnames = locationchoice.add_fnames(fnames,est_params)
    if show: print misc.resultstotable(fnames,results)
    misc.resultstocsv(fit,fnames,results,tmp_outcsv,tblname=tmp_outtitle)
    dset.store_coeff(tmp_coeffname,zip(*results)[0],fnames)

  print "Finished executing in %f seconds" % (time.time()-t1)

############
# SIMULATION
############

def simulate(dset,config,year,sample_rate=.05,variables=None,show=False):

  t1 = time.time()
  choosers = fetch_table(dset,config,simulate=1)
  
  output_csv, output_title, coeff_name, output_varname = config["output_names"]
  
  assert 'dep_var' in config
  dep_var = config['dep_var']
 
  if 'relocation_rates' in config:
    reloc_cfg = config['relocation_rates']
    assert "rate_table" in reloc_cfg and "rate_field" in reloc_cfg
    rate_table = eval(reloc_cfg['rate_table'])
    rate_field = reloc_cfg['rate_field']
    movers = dset.relocation_rates(choosers,rate_table,rate_field)
    choosers[dep_var].ix[movers] = -1
    # add current unplaced
    movers = choosers[choosers[dep_var]==-1]

  else: movers = choosers # everyone moves

  print "Total new agents and movers = %d" % len(movers.index)

  assert 'alternatives' in config
  alternatives = eval(config['alternatives'])

  lotterychoices = False
  if 'supply_constraint' in config:
    empty_units = eval(config['supply_constraint'])
    if "demand_amount_scale" in config: empty_units /=  float(config["demand_amount_scale"])
    empty_units = empty_units[empty_units>0].order(ascending=False)
    if 'dontexpandunits' in config and config['dontexpandunits'] == True: 
      alternatives = alternatives.ix[empty_units.index]
      alternatives["supply"] = empty_units
      lotterychoices = True
    else: 
      alternatives = alternatives.ix[np.repeat(empty_units.index,empty_units.values.astype('int'))]
    print "There are %s empty units in %s locations total in the region" % (empty_units.sum(),len(empty_units))

  alternatives = merge(dset,alternatives,config)

  print "Finished specifying model in %f seconds" % (time.time()-t1)

  t1 = time.time()

  pdf = pd.DataFrame(index=alternatives.index) 
  segments = [(None,movers)]
  if 'segment' in config:
    for varname in config['segment']:
      if varname not in movers.columns:
        movers[varname] = calcvar(movers,config,dset,varname)
    segments = movers.groupby(config['segment'])

  for name, segment in segments:

    segment = segment.head(1)

    name = str(name)
    if name is not None: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv%name, output_title%name, coeff_name%name
    else: tmp_outcsv, tmp_outtitle, tmp_coeffname = output_csv, output_title, coeff_name
  
    SAMPLE_SIZE = alternatives.index.size # don't sample
    sample, alternative_sample, est_params = \
             locationchoice.mnl_interaction_dataset(segment,alternatives,SAMPLE_SIZE,chosenalts=None)
    data = spec(alternative_sample,config)
    data = data.as_matrix()

    coeff = dset.load_coeff(tmp_coeffname)
    probs = locationchoice.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
    pdf['segment%s'%name] = pd.Series(probs.flatten(),index=alternatives.index)

  print "Finished creating pdf in %f seconds" % (time.time()-t1)
  if len(pdf.columns): print pdf.describe()
  t1 = time.time()

  if 'supply_constraint' in config: # draw from actual units
    new_homes = pd.Series(np.ones(len(movers.index))*-1,index=movers.index)
    mask = np.zeros(len(alternatives.index),dtype='bool')
    for name, segment in segments:
      name = str(name)
      print "Assigning units to %d agents of segment %s" % (len(segment.index),name)
      p=pdf['segment%s'%name].values
     
      def choose(p,mask,alternatives,segment,new_homes,minsize=None):
        p = copy.copy(p)

        if minsize is not None: p[alternatives.supply<minsize] = 0
        else: p[mask] = 0 # already chosen
        print "Choosing from %d nonzero alts" % np.count_nonzero(p)

        try: 
          indexes = np.random.choice(len(alternatives.index),len(segment.index),replace=False,p=p/p.sum())
        except:
          print "WARNING: not enough options to fit agents, will result in unplaced agents"
          return
        new_homes.ix[segment.index] = alternatives.index.values[indexes]
        
        if minsize is not None: alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
        else: mask[indexes] = 1
        
        return mask,new_homes

      if lotterychoices and "demand_amount" not in config:
        print "WARNING: you've specified a supply constraint but no demand_amount - all demands will be of value 1"

      if lotterychoices and "demand_amount" in config:
          
        tmp = segment[config["demand_amount"]]
        if "demand_amount_scale" in config: tmp /= float(config["demand_amount_scale"])

        for name, subsegment in reversed(list(segment.groupby(tmp.astype('int')))):
          
          print "Running subsegment with size = %s, num agents = %d" % (name, len(subsegment.index))
          mask,new_homes = choose(p,mask,alternatives,subsegment,new_homes,minsize=int(name))
      
      else:  mask,new_homes = choose(p,mask,alternatives,segment,new_homes)

    build_cnts = new_homes.value_counts()
    print "Assigned %d agents to %d locations with %d unplaced" % \
                      (new_homes.size,build_cnts.size,build_cnts.get(-1,0))

    table = eval(config['table']) # need to go back to the whole dataset
    table[dep_var].ix[new_homes.index] = new_homes.values.astype('int32')
    if output_varname: dset.store_attr(output_varname,year,copy.deepcopy(table[dep_var]))

  if 'web_service' in config and config['web_service']: # results to web service
    parcel_predictions = pd.merge(dset.fetch('parcels')[['_node_id']],
                                            pdf*alternatives.index.size,left_on='_node_id',right_index=True)
    from urbansimd import urbansimd
    PORT = 8759
    parcelcentroids = pd.read_csv(os.path.join(misc.data_dir(),'parcel_centroids.csv'),index_col='parcel_id')
    tabledict = {'location_results':parcel_predictions.astype('float')}
    urbansimd.start_service(parcelcentroids,tabledict,PORT,srid=3857,sqlite=0)
  
  print "Finished assigning agents in %f seconds" % (time.time()-t1)

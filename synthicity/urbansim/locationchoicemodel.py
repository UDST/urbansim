{% from 'modelspec.py' import IMPORTS, MERGE, SPEC, CALCVAR, TABLE with context %}
{{ IMPORTS() }}
SAMPLE_SIZE=100

##############
#  ESTIMATION
##############

{% if template_mode == "estimate" %}
def {{modelname}}_estimate(dset,year=None,show=True):

  assert "{{model}}" == "locationchoicemodel" # should match!
  returnobj = {}
  
  # TEMPLATE configure table
  {{ TABLE("choosers")|indent(2) }}
  # ENDTEMPLATE
  
  {% if est_sample_size -%} 
  # TEMPLATE randomly choose estimatiors
  choosers = choosers.ix[np.random.choice(choosers.index, {{est_sample_size}},replace=False)]
  # ENDTEMPLATE
  {% endif -%}
  
  # TEMPLATE specifying alternatives
  alternatives = {{alternatives}}
  # ENDTEMPLATE
  
  {% if merge -%}
  # TEMPLATE merge
  {{- MERGE("alternatives",merge) | indent(2) }}
  # ENDTEMPLATE
  {% endif -%}

  t1 = time.time()

  {% if segment is not defined -%}
  segments = [(None,choosers)]
  {% else -%}
  # TEMPLATE creating segments
  {% for varname in segment -%}
  {% if varname in var_lib -%}
  
  if "{{varname}}" not in choosers.columns: 
    choosers["{{varname}}"] = {{CALCVAR("choosers",varname,var_lib)}}
  {% endif -%}
  {% endfor -%}
  segments = choosers.groupby({{segment}})
  # ENDTEMPLATE
  {% endif  %}
  
  for name, segment in segments:

    name = str(name)
    outname = "{{modelname}}" if name is None else "{{modelname}}_"+name

    global SAMPLE_SIZE
    {% if alt_sample_size -%}
    SAMPLE_SIZE = {{alt_sample_size}}
    {% endif -%}
    sample, alternative_sample, est_params = interaction.mnl_interaction_dataset(
                                        segment,alternatives,SAMPLE_SIZE,chosenalts=segment["{{dep_var}}"])

    print "Estimating parameters for segment = %s, size = %d" % (name, len(segment.index)) 

    # TEMPLATE computing vars
    {{ SPEC("alternative_sample","data",submodel="name") | indent(4) }}
    # ENDTEMPLATE
    if show: print data.describe()

    d = {}
    d['columns'] = fnames = data.columns.tolist()

    data = data.as_matrix()
    if np.amax(data) > 500.0:
      print "WARNING: the max value in this estimation data is large, it's likely you need to log transform the input"
    fit, results = interaction.estimate(data,est_params,SAMPLE_SIZE)
 
    fnames = interaction.add_fnames(fnames,est_params)
    if show: print misc.resultstotable(fnames,results)
    misc.resultstocsv(fit,fnames,results,outname+"_estimate.csv",tblname=outname)
    
    d['null loglik'] = float(fit[0])
    d['converged loglik'] = float(fit[1])
    d['loglik ratio'] = float(fit[2])
    d['est_results'] = [[float(x) for x in result] for result in results]
    returnobj[name] = d
    
    dset.store_coeff(outname,zip(*results)[0],fnames)

  print "Finished executing in %f seconds" % (time.time()-t1)
  return returnobj

{% else %}
############
# SIMULATION
############

def {{modelname}}_simulate(dset,year=None,show=True):

  returnobj = {}

  t1 = time.time()
  # TEMPLATE configure table
  {{ TABLE("choosers")|indent(2) }}
  # ENDTEMPLATE
  
  # TEMPLATE dependent variable
  depvar = "{{dep_var}}"
  # ENDTEMPLATE

  {% if relocation_rates -%} 
  # TEMPLATE computing relocations
  movers = dset.relocation_rates(choosers,{{relocation_rates.rate_table}},"{{relocation_rates.rate_field}}")
  choosers["{{dep_var}}"].ix[movers] = -1
  # add current unplaced
  movers = choosers[choosers["{{dep_var}}"]==-1]
  # ENDTEMPLATE
  {% else -%}
  movers = choosers # everyone moves
  {% endif %}

  print "Total new agents and movers = %d" % len(movers.index)

  # TEMPLATE specifying alternatives
  alternatives = {{alternatives}}
  # ENDTEMPLATE
  
  lotterychoices = False
  {% if supply_constraint -%}
  # TEMPLATE computing supply constraint
  empty_units = {{supply_constraint}}
  {% if demand_amount_scale -%}
  empty_units /=  float({{demand_amount_scale}})
  {% endif -%}
  empty_units = empty_units[empty_units>0].order(ascending=False)
  {% if dontexpandunits -%} 
  alternatives = alternatives.ix[empty_units.index]
  alternatives["supply"] = empty_units
  lotterychoices = True
  {% else -%}  
  alternatives = alternatives.ix[np.repeat(empty_units.index,empty_units.values.astype('int'))].reset_index()
  {% endif -%}
  print "There are %s empty units in %s locations total in the region" % (empty_units.sum(),len(empty_units))
  # ENDTEMPLATE
  {% endif -%}

  {% if merge -%}
  # TEMPLATE merge
  {{- MERGE("alternatives",merge) | indent(2) }}
  # ENDTEMPLATE
  {% endif %}

  print "Finished specifying model in %f seconds" % (time.time()-t1)

  t1 = time.time()

  pdf = pd.DataFrame(index=alternatives.index) 
  {% if segment is not defined -%}
  segments = [(None,movers)]
  {% else %}
  # TEMPLATE creating segments
  {% for varname in segment -%}
  {% if varname in var_lib -%}
  
  if "{{varname}}" not in movers.columns: 
    movers["{{varname}}"] = {{CALCVAR("movers",varname,var_lib)}}
  {% endif -%}
  {% endfor -%}
  segments = movers.groupby({{segment}})
  # ENDTEMPLATE
  {% endif  %}

  for name, segment in segments:

    segment = segment.head(1)

    name = str(name)
    outname = "{{modelname}}" if name is None else "{{modelname}}_"+name
  
    SAMPLE_SIZE = alternatives.index.size # don't sample
    sample, alternative_sample, est_params = \
             interaction.mnl_interaction_dataset(segment,alternatives,SAMPLE_SIZE,chosenalts=None)
    # TEMPLATE computing vars
    {{ SPEC("alternative_sample","data",submodel="name") | indent(4) }}
    # ENDTEMPLATE
    data = data.as_matrix()

    coeff = dset.load_coeff(outname)
    probs = interaction.mnl_simulate(data,coeff,numalts=SAMPLE_SIZE,returnprobs=1)
    pdf['segment%s'%name] = pd.Series(probs.flatten(),index=alternatives.index) 

  print "Finished creating pdf in %f seconds" % (time.time()-t1)
  if len(pdf.columns) and show: print pdf.describe()
  returnobj[name] = misc.pandasdfsummarytojson(pdf.describe(),ndigits=10)
  pdf.describe().to_csv(os.path.join(misc.output_dir(),"{{modelname}}_simulate.csv"))
    
  {% if save_pdf -%}
  dset.save_tmptbl("{{save_pdf}}",pdf)
  {% endif %}

  {%- if supply_constraint -%}
  t1 = time.time()
  # draw from actual units
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
        return mask,new_homes
      new_homes.ix[segment.index] = alternatives.index.values[indexes]
        
      if minsize is not None: alternatives["supply"].ix[alternatives.index.values[indexes]] -= minsize
      else: mask[indexes] = 1
        
      return mask,new_homes

    if lotterychoices and {{demand_amount == None}}:
      print "WARNING: you've specified a supply constraint but no demand_amount - all demands will be of value 1"

    if lotterychoices and {{demand_amount != None}}:
          
      tmp = segment["{{demand_amount}}"]
      {% if demand_amount_scale -%}
      tmp /= {{demand_amount_scale}}
      {% endif %}

      for name, subsegment in reversed(list(segment.groupby(tmp.astype('int')))):
          
        print "Running subsegment with size = %s, num agents = %d" % (name, len(subsegment.index))
        mask,new_homes = choose(p,mask,alternatives,subsegment,new_homes,minsize=int(name))
      
    else:  mask,new_homes = choose(p,mask,alternatives,segment,new_homes)

  build_cnts = new_homes.value_counts()
  print "Assigned %d agents to %d locations with %d unplaced" % \
                      (new_homes.size,build_cnts.size,build_cnts.get(-1,0))

  table = {{table_sim if table_sim else table}} # need to go back to the whole dataset
  table["{{dep_var}}"].ix[new_homes.index] = new_homes.values.astype('int32')
  dset.store_attr("{{output_varname}}",year,copy.deepcopy(table["{{dep_var}}"]))
  print "Finished assigning agents in %f seconds" % (time.time()-t1)
  {% endif -%}

  return returnobj

{% endif %}

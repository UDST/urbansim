from synthicity.utils import texttable as tt
from synthicity.urbansim import interaction
import os, sys, getopt, csv, string, time, json
import pandas as pd, numpy as np

def run_model(fname,dset,show=1,estimate=1,year=2010,variables=None):
  from synthicity.urbansim import hedonicmodel, locationchoicemodel, minimodel, transitionmodel, networks
  if type(fname) == type(""): 
    print "Running %s\n" % fname
    config = json.loads(open(fname).read()) 
  else: 
    config = fname
    if "_id" in config: fname = config["_id"]
    elif "name" in config: fname = config["name"]
    else: fname = "tmp_model"
  assert "model" in config
  model = eval(config["model"])
  if estimate: model.estimate(dset,config,2010,show=show,variables=variables)
  if config["model"] == "hedonicmodel" or not estimate: 
    t1 = time.time()
    model.simulate(dset,config,year,show=show,variables=variables)
    print "SIMULATED %s model in %.3f seconds" % (fname,time.time()-t1)

VARNAMESDICT = {
'number_of_stories':'Stories',
'rentable_building_area':'Area',
'nets_all_regional1_30':'Median accessibility',
'nets_all_regional2_30':'Travel time unreliability',
'const':'Constant',
'totsum':'Area',
'weightedrent':'Rent',
'retpct':'Percent retail',
'indpct':'Percent industrial',
'SQft':'Area',
'Lot size':'Lot size',
'historic':'Historic',
'new':'New',
'AvgOfSquareFeet':'Area',
'demo_average_income_average_local':'Average income',
'const':'Constant',
'rent':'Monthly rent',
'residential_units':'Number of units',
'hoodrenters':'Area renters',
'noderenters':'Node renters',
'sales_price':'Sales price',
'sales_price x income':'Sales price $\\times$ income',
'demo_averageincome_local':'Average income',
'income x income':'Interacted income',
}

def data_dir(): return os.path.join(os.environ['DATA_HOME'],'data')
def runs_dir(): return os.path.join(os.environ['DATA_HOME'],'runs')
def output_dir(): return os.path.join(os.environ['DATA_HOME'],'output')

def get_run_number():
    try:
      f = open(os.path.join(os.environ['DATA_HOME'],'RUNNUM'),'r')
      num = int(f.read())
      f.close()
    except:
      num = 0
    f = open(os.path.join(os.environ['DATA_HOME'],'RUNNUM'),'w')
    f.write(str(num+1))
    f.close()
    return num

# conver significance to text representation like R does
def signif(val):
    val = abs(val)
    if val > 3.1: return '***'
    elif val > 2.33: return '**'
    elif val > 1.64: return '*'
    elif val > 1.28: return '.'
    return ''

def maketable(fnames,results,latex=0):
    results = [[string.replace(x,'_',' ')]+['%.2f'%float(z) for z in list(y)]+
                              [signif(y[-1])] for x, y in zip(fnames,results)]
    if latex: return [['Variables','$\\beta$','$\\sigma$','\multicolumn{1}{c}{T-score}','Significance']]+results
    return [['Variables','Coefficient','Stderr','T-score','Significance']]+results

LATEX_TEMPLATE = """
\\begin{table}\label{%(tablelabel)s}
\caption { %(tablename)s }
\\begin{center}
    \\begin{tabular}{lcc S[table-format=3.2] c}
                %(tablerows)s
                \hline
                %(metainfo)s

    \end{tabular}
\end{center}
\end{table}
"""

TABLENUM = 0
def resultstolatex(fit,fnames,results,filename,hedonic=0,tblname=None):
    global TABLENUM
    TABLENUM += 1
    filename = filename + '.tex'
    results = maketable(fnames,results,latex=1)
    f = open(os.path.join(output_dir(),filename),'w')
    tablerows = ''
    for row in results:
        tablerows += string.join(row,sep='&') + '\\\\\n'
        if row == results[0]: tablerows += '\hline\n'
    if hedonic: fitnames = ['R$^2$','Adj-R$^2$']
    else: fitnames = ['Null loglik','Converged loglik','Loglik ratio']
    metainfo = ''
    for t in zip(fitnames,fit):
        metainfo += '%s %.2f &&&&\\\\\n' % t
    
    data = {'tablename':tblname,'tablerows':tablerows,'metainfo':metainfo,'tablelabel':'table%d'%TABLENUM}
    f.write(LATEX_TEMPLATE%data)
    f.close()

def resultstocsv(fit,fnames,results,filename,hedonic=0,tolatex=1,tblname=None):
    fnames = [VARNAMESDICT.get(x,x) for x in fnames]
    if tolatex: resultstolatex(fit,fnames,results,filename,hedonic,tblname=tblname)
    results = maketable(fnames,results)
    f = open(os.path.join(output_dir(),filename),'w')
    if hedonic: f.write('R-squared,%f\nAdj-R-squared,%f\n\n\n'%fit)
    else: f.write('null loglik,%f\nconverged loglik,%f\nloglik ratio,%f\n\n\n'%fit)
    csvf = csv.writer(f,lineterminator='\n')
    for row in results: csvf.writerow(row)
    f.close()

def resultstotable(fnames,results):
    results = maketable(fnames,results)

    tab = tt.Texttable()
    tab.add_rows(results)
    tab.set_cols_align(['r','r','r','r','l'])

    return tab.draw()

RUNCONFIG = {}
def process_args():
    opts, args = getopt.getopt(sys.argv[1:], "glnd")
    for o, a in opts:
      if o == "-g": interaction.enable_gpu()
      if o == "-n": RUNCONFIG['update_networks'] = 1
      if o == "-d": RUNCONFIG['run_developer'] = 1
      if o == "-l": RUNCONFIG['lottery_choices'] = 1
    return args

naics_d = { \
  11: 'Agriculture', 
  21: 'Mining',
  22: 'Utilities',
  23: 'Construction',
  31: 'Manufacturing',
  32: 'Manufacturing',
  33: 'Manufacturing',
  42: 'Wholesale',
  44: 'Retail',
  45: 'Retail',
  48: 'Transportation',
  49: 'Warehousing',
  51: 'Information',
  52: 'Finance and Insurance',
  53: 'Real Estate',
  54: 'Professional',
  55: 'Management',
  56: 'Administrative',
  61: 'Educational',
  62: 'Health Care',
  71: 'Arts',
  72: 'Accomodation and Food',
  81: 'Other',
  92: 'Public',
  99: 'Unknown'
}

def naicsname(val):
    return naics_d[val]

def writenumpy(series,name,outdir="."):
    #print name,series.describe()
    series = series.dropna()
    series.sort()
    fname = os.path.join(outdir,'tmp' if not name else name)
    np.savez(fname,parcel_id=series.index.values.astype('int32'),values=series.values.astype('float32'))

def writenumpy_df(df,outdir="."):
    for column in df.columns:
        writenumpy(df[column],column,outdir)

def numpymat2df(mat):
    return pd.DataFrame(dict(('x%d'%i,mat[:,i]) for i in range(mat.shape[1])))

def df64bitto32bit(tbl):
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in tbl.columns:
        if tbl[colname].dtype == np.float64: newtbl[colname] = tbl[colname].astype('float32') 
        elif tbl[colname].dtype == np.int64: newtbl[colname] = tbl[colname].astype('int32')
        else: newtbl[colname] = tbl[colname]
    return newtbl

def series64bitto32bit(s):
    if s.dtype == np.float64: return s.astype('float32') 
    elif s.dtype == np.int64: return s.astype('int32')
    return s

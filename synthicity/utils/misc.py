from synthicity.utils import texttable as tt
from synthicity.urbanchoice import interaction
import os, copy, sys, getopt, csv, string, time, json
import pandas as pd, numpy as np
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict

# these are the lists of modes available for each model
MODES_D = defaultdict(lambda: ["estimate","simulate"], {
 "minimodel": ["run"],
 "modelset": ["run"],
 "networks":  ["run"] 
})

# generate a model from a json file, will do so for all modes listed above
def gen_model(configname):
  def droptable(d):
    d = copy.copy(d)
    del d['table']
    return d
  # look for templates in PYTHONPATH
  templatedirs = filter(os.path.isdir,[os.path.join(x,'synthicity','urbansim') for x in sys.path])
  
  j2_env = Environment(loader=FileSystemLoader(templatedirs),trim_blocks=True)
  j2_env.filters['droptable'] = droptable

  config = json.loads(open(configname).read())
  if 'model' not in config: 
    print "Not generating %s" % configname
    return {}
  model = config['model']
  d = {}
  for mode in MODES_D[model]:
    
    basename = os.path.splitext(os.path.basename(configname))[0]
    dirname = os.path.dirname(configname)
    print "Running %s with mode %s" % (basename,mode)

    if 'var_lib_file' in config:
      var_lib = json.loads(open(os.path.join(dirname,config['var_lib_file'])).read()) 
      config["var_lib"] = dict(var_lib.items()+config.get("var_lib",{}).items())
    
    config['modelname'] = basename
    config['template_mode'] = mode
    d[mode] = j2_env.get_template(model+'.py').render(**config)
  
  return d


def data_dir(): return os.path.join(os.environ['DATA_HOME'],'data')
def models_dir(): return os.path.join(os.environ['DATA_HOME'],'models')
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

# create a table, either latex of csv or text
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

# override this to modify variable names for publication
VARNAMESDICT = {}

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

# this is an ascii table for the terminal
def resultstotable(fnames,results):
    results = maketable(fnames,results)

    tab = tt.Texttable()
    tab.add_rows(results)
    tab.set_cols_align(['r','r','r','r','l'])

    return tab.draw()

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

# just used to reduce size
def df64bitto32bit(tbl):
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in tbl.columns:
        if tbl[colname].dtype == np.float64: newtbl[colname] = tbl[colname].astype('float32') 
        elif tbl[colname].dtype == np.int64: newtbl[colname] = tbl[colname].astype('int32')
        else: newtbl[colname] = tbl[colname]
    return newtbl

# also to reduce size
def series64bitto32bit(s):
    if s.dtype == np.float64: return s.astype('float32') 
    elif s.dtype == np.int64: return s.astype('int32')
    return s

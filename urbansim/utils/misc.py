from __future__ import print_function

import csv
import os
import string
import yaml

import numpy as np
import pandas as pd

from urbansim.utils import texttable as tt


def mkifnotexists(folder):
    d = os.path.join(os.getenv('DATA_HOME', "."), folder)
    if not os.path.exists(d):
        os.mkdir(d)
    return d


def data_dir():
    return mkifnotexists("data")


def models_dir():
    return mkifnotexists("models")


def configs_dir():
    return mkifnotexists("configs")


def charts_dir():
    return mkifnotexists("charts")


def maps_dir():
    return mkifnotexists("maps")


def runs_dir():
    return mkifnotexists("runs")


def reports_dir():
    return mkifnotexists("reports")


def coef_dir():
    return mkifnotexists("coeffs")


def output_dir():
    return mkifnotexists("output")


def debug_dir():
    return mkifnotexists("debug")


def config(fname):
    return os.path.join(configs_dir(), fname)


def ordered_yaml(cfg):
    order = ["name", "model_type", "fit_filters", "predict_filters",
                     "choosers_fit_filters", "choosers_predict_filters",
                     "alts_fit_filters", "alts_predict_filters",
                     "interaction_predict_filters",
                     "choice_column", "sample_size", "estimation_sample_size",
                     "patsy", "dep_var", "dep_var_transform", "model_expression",
                     "ytransform"]

    s = ""
    for key in order:
        if key not in cfg:
            continue
        s += yaml.dump({key: cfg[key]}, default_flow_style=False, indent=4)
        s += "\n"

    for key in cfg:
        if key in order:
            continue
        s += yaml.dump({key: cfg[key]}, default_flow_style=False, indent=4)
        s += "\n"

    return s


def make_model_expression(cfg):
    """
    Turn the parameters into the string expected by patsy.

    Parameters
    ----------
    cfg : A dictionary of key-value pairs.  'patsy' defines
        patsy variables, 'dep_var' is the dependent variable,
        and 'dep_var_transform' is the transformation of the
        dependent variable.

    Returns
    -------
    Modifies the dictionary of params in place
    """
    if "patsy" not in cfg:
        return
    if "dep_var" not in cfg:
        return
    if "dep_var_transform" not in cfg:
        return
    patsy_exp = cfg['patsy']
    if type(patsy_exp) == list:
        patsy_exp = ' + '.join(cfg['patsy'])
    # deal with missing dep_var_transform
    patsy_exp = '%s(%s) ~ ' % (
        cfg['dep_var_transform'], cfg['dep_var']) + patsy_exp
    cfg['model_expression'] = patsy_exp


def get_run_number():
    try:
        f = open(os.path.join(os.getenv('DATA_HOME', "."), 'RUNNUM'), 'r')
        num = int(f.read())
        f.close()
    except:
        num = 1
    f = open(os.path.join(os.getenv('DATA_HOME', "."), 'RUNNUM'), 'w')
    f.write(str(num + 1))
    f.close()
    return num


# conver significance to text representation like R does
def signif(val):
    val = abs(val)
    if val > 3.1:
        return '***'
    elif val > 2.33:
        return '**'
    elif val > 1.64:
        return '*'
    elif val > 1.28:
        return '.'
    return ''


# create a table, either latex of csv or text
def maketable(fnames, results, latex=False):
    results = [[string.replace(x, '_', ' ')] +
               ['%.2f' % float(z) for z in list(y)] +
               [signif(y[-1])] for x, y in zip(fnames, results)]
    if latex:
        return [['Variables', '$\\beta$', '$\\sigma$',
                 '\multicolumn{1}{c}{T-score}', 'Significance']] + results
    return [['Variables', 'Coefficient', 'Stderr', 'T-score',
             'Significance']] + results

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


def resultstolatex(fit, fnames, results, filename, hedonic=0, tblname=None):
    global TABLENUM
    TABLENUM += 1
    filename = filename + '.tex'
    results = maketable(fnames, results, latex=1)
    f = open(os.path.join(debug_dir(), filename), 'w')
    tablerows = ''
    for row in results:
        tablerows += string.join(row, sep='&') + '\\\\\n'
        if row == results[0]:
            tablerows += '\hline\n'
    if hedonic:
        fitnames = ['R$^2$', 'Adj-R$^2$']
    else:
        fitnames = ['Null loglik', 'Converged loglik', 'Loglik ratio']
    metainfo = ''
    for t in zip(fitnames, fit):
        metainfo += '%s %.2f &&&&\\\\\n' % t

    data = {'tablename': tblname, 'tablerows': tablerows,
            'metainfo': metainfo, 'tablelabel': 'table%d' % TABLENUM}
    f.write(LATEX_TEMPLATE % data)
    f.close()

# override this to modify variable names for publication
VARNAMESDICT = {}


def resultstocsv(fit, fnames, results, filename, hedonic=False, tolatex=True,
                 tblname=None):
    fnames = [VARNAMESDICT.get(x, x) for x in fnames]
    if tolatex:
        resultstolatex(
            fit, fnames, results, filename, hedonic, tblname=tblname)
    results = maketable(fnames, results)
    f = open(os.path.join(output_dir(), filename), 'w')
    csvf = csv.writer(f, lineterminator='\n')
    for row in results:
        csvf.writerow(row)
    f.close()


# this is an ascii table for the terminal
def resultstotable(fnames, results):
    results = maketable(fnames, results)

    tab = tt.Texttable()
    tab.add_rows(results)
    tab.set_cols_align(['r', 'r', 'r', 'r', 'l'])

    return tab.draw()

naics_d = {
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


def writenumpy(series, name, outdir="."):
    # print name,series.describe()
    series = series.dropna()
    series.sort()
    fname = os.path.join(outdir, 'tmp' if not name else name)
    np.savez(fname, parcel_id=series.index.values.astype(
        'int32'), values=series.values.astype('float32'))


def writenumpy_df(df, outdir="."):
    for column in df.columns:
        writenumpy(df[column], column, outdir)


def numpymat2df(mat):
    return pd.DataFrame(
        dict(('x%d' % i, mat[:, i]) for i in range(mat.shape[1])))


# just used to reduce size
def df64bitto32bit(tbl):
    newtbl = pd.DataFrame(index=tbl.index)
    for colname in tbl.columns:
        if tbl[colname].dtype == np.float64:
            newtbl[colname] = tbl[colname].astype('float32')
        elif tbl[colname].dtype == np.int64:
            newtbl[colname] = tbl[colname].astype('int32')
        else:
            newtbl[colname] = tbl[colname]
    return newtbl


# also to reduce size
def series64bitto32bit(s):
    if s.dtype == np.float64:
        return s.astype('float32')
    elif s.dtype == np.int64:
        return s.astype('int32')
    return s


def pandassummarytojson(v, ndigits=3):
    return {i: round(float(v.ix[i]), ndigits) for i in v.index}


def pandasdfsummarytojson(df, ndigits=3):
    df = df.transpose()
    return {k: pandassummarytojson(v, ndigits) for k, v in df.iterrows()}

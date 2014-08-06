import cPickle
import yaml

import numpy as np
import pandas as pd
import logging

from . import misc
from ..models import util
import urbansim.sim.simulation as sim

logger = logging.getLogger(__name__)

NETWORKS = None


def from_yaml(cfgname):
    print "Computing accessibility variables"
    cfg = yaml.load(open(misc.config(cfgname)))

    nodes = pd.DataFrame(index=NETWORKS.external_nodeids)

    node_col = cfg.get('node_col', None)

    for variable in cfg['variable_definitions']:

        name = variable["name"]
        print "Computing %s" % name

        decay = {
            "exponential": "DECAY_EXP",
            "linear": "DECAY_LINEAR",
            "flat": "DECAY_FLAT"
        }.get(variable.get("decay", "linear"))

        agg = {
            "sum": "AGG_SUM",
            "average": "AGG_AVE",
            "stddev": "AGG_STDDEV"
        }.get(variable.get("aggregation", "sum"))

        vname = variable.get("varname", None)

        radius = variable["radius"]

        dfname = variable["dataframe"]

        flds = [vname] if vname else []
        if 'add_fields' in variable:
            flds += variable['add_fields']
        if node_col:
            flds.append(node_col)
        logger.info("    Fields available to accvar =", ', '.join(flds))

        df = sim.get_table(dfname).to_frame(flds)

        if "filters" in variable:
            df = util.apply_filter_query(df, variable["filters"])
            logger.info("    Filters = %s" % variable["filters"])

        logger.info("    dataframe = %s, varname=%s" % (dfname, vname))
        logger.info("    radius = %s, aggregation = %s, decay = %s" % (
            radius, agg, decay))

        nodes[name] = NETWORKS.accvar(
            df, radius, node_ids=node_col, agg=agg, decay=decay,
            vname=vname).astype('float').values

        if "apply" in variable:
            nodes[name] = nodes[name].apply(eval(variable["apply"]))

    return nodes


class Networks:

    # flatten_nodeids is used when there is one graph to make a list of nodeids
    # rather than a list of lists - it doesn't work right now unfortunately
    def __init__(self, filenames, factors, maxdistances, twoway,
                 impedances=None, flatten_nodeids=False):
        if not filenames:
            return
        from pyaccess.pyaccess import PyAccess
        self.pya = PyAccess()
        self.pya.createGraphs(len(filenames))
        if impedances is None:
            impedances = [None] * len(filenames)
        self.nodeids = []
        self.external_nodeids = []
        for num, filename, factor, maxdistance, twoway, impedance in \
                zip(range(len(filenames)), filenames, factors, maxdistances,
                    twoway, impedances):
            net = cPickle.load(open(filename))
            if impedance is None:
                impedance = "net['edgeweights']"
            impedance = eval(impedance)
            self.pya.createGraph(
                num, net['nodeids'], net['nodes'], net['edges'],
                impedance * factor, twoway=twoway)
            if len(filenames) == 1 and flatten_nodeids:
                self.nodeids = net['nodeids']
            else:
                # these are the internal ids
                self.nodeids += zip([num] * len(net['nodeids']),
                                    range(len(net['nodeids'])))
            self.external_nodeids.append(net['nodeids'])
            self.pya.precomputeRange(maxdistance, num)

    def accvar(self, df, distance, node_ids=None, xname='x', yname='y',
               vname=None, agg="AGG_SUM", decay="DECAY_LINEAR"):
        assert self.pya  # need to generate pyaccess first
        pya = self.pya
        if isinstance(agg, str):
            agg = getattr(pya, agg)
        if isinstance(decay, str):
            decay = getattr(pya, decay)
        if vname:
            df = df.dropna(subset=[vname])
        if node_ids is None:
            xys = np.array(df[[xname, yname]], dtype="float32")
            node_ids = []
            for gno in range(pya.numgraphs):
                node_ids.append(pya.XYtoNode(xys, distance=1000, gno=gno))
        if isinstance(node_ids, str):
            l = len(df)
            df = df.dropna(subset=[node_ids])
            newl = len(df)
            if newl-l > 0:
                print "Removed %d rows because there are no node_ids" % (newl-l)
            node_ids = [df[node_ids].astype("int32").values]
        elif not isinstance(node_ids, list):
            node_ids = [node_ids]

        pya.initializeAccVars(1)
        num = 0
        aggvar = df[vname].astype('float32') if vname is not None else np.ones(
            len(df.index), dtype='float32')
        pya.initializeAccVar(num, node_ids, aggvar, preaggregate=0)
        res = []
        for gno in range(pya.numgraphs):
            res.append(pya.getAllAggregateAccessibilityVariables(
                distance, num, agg, decay, gno=gno))
        return pd.Series(
            np.concatenate(res), index=pd.MultiIndex.from_tuples(self.nodeids))

    def addnodeid(self, df):

        try:
            xys = np.array(df[['x', 'y']], dtype="float32")
        except:
            xys = np.array(df[['X', 'Y']], dtype="float32")

        for gno in range(self.pya.numgraphs):
            df['_node_id%d' % gno] = pd.Series(
                self.pya.XYtoNode(xys, gno=gno), index=df.index)
        # assign the external id as well
        df['_node_id'] = pd.Series(self.pya.getGraphIDS()[df['_node_id0'].values],
                                   index=df.index)
        return df

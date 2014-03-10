import cPickle
import os
import sys
import time

import numpy as np
import pandas as pd

from synthicity.utils import misc

NETWORKS = None


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
        if node_ids is None:
            xys = np.array(df[[xname, yname]], dtype="float32")
            node_ids = []
            for gno in range(pya.numgraphs):
                node_ids.append(pya.XYtoNode(xys, distance=1000, gno=gno))
        if isinstance(node_ids, str):
            node_ids = [df[node_ids].values]
        elif not isinstance(node_ids, []):
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

    def addnodeid(self, df, tosrid=0):

        try:
            xys = np.array(df[['x', 'y']], dtype="float32")
        except:
            xys = np.array(df[['X', 'Y']], dtype="float32")

        if tosrid:  # convert coordinate system
            from synthicity.utils import geomisc
            print "Converting srids (potentially takes a long time)"
            xys = geomisc.np_coord_convert_all(xys)
            print "Finished conversion"

        for gno in range(self.pya.numgraphs):
            df['_node_id%d' % gno] = pd.Series(
                self.pya.XYtoNode(xys, gno=gno), index=df.index)
        # assign the external id as well
        df['_node_id'] = pd.Series(
            self.pya.getGraphIDS()[df['_node_id0']], index=df.index)
        return df

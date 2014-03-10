import time

import numpy as np
import pandas as pd

from synthicity.urbansim import interaction, mnl
from synthicity.utils import misc

GPU = 0
EMTOL = 1e-06
MAXITER = 10000


def prep_cm_data(cmdata, numclasses, cmfnames):
    numobs, numvars = cmdata.shape
    newcmfnames = []
    newcmdata = np.zeros((numobs * numclasses, numvars * (numclasses - 1)))
    for i in range(cmdata.shape[0]):
        for j in range(1, numclasses):
            newcmdata[
                i * numclasses + j, (j - 1) * numvars:j * numvars] = cmdata[i]
    for j in range(1, numclasses):
        for i in range(cmdata.shape[1]):
            newcmfnames.append(cmfnames[i] + ', cls%d' % j)
    return newcmdata, newcmfnames


def lcmnl_estimate(cmdata, numclasses, csdata, numalts, chosen,
                   maxiter=MAXITER, emtol=EMTOL, skipprep=False, csbeta=None,
                   cmbeta=None, csfnames=None, cmfnames=None):

    loglik = -999999
    l_0 = None
    if csbeta is None:
        csbeta = [np.random.rand(csdata.shape[1]) for i in range(numclasses)]
    if csfnames is None:
        csfnames = ['cs%d' % i for i in range(csdata.shape[1])]
    if cmfnames is None:
        cmfnames = ['cm%d' % i for i in range(cmdata.shape[1])]
    if not skipprep:
        cmdata, cmfnames = prep_cm_data(cmdata, numclasses, cmfnames)
    if cmbeta is None:
        cmbeta = np.random.rand(cmdata.shape[1]) * 10.0 - 5.0
    results_d = {}

    for i in range(maxiter):
        print "Running iteration %d" % (i + 1)
        print time.ctime()

        # EXPECTATION
        def expectation(cmbeta, csbeta):
            print "Running class membership model"
            cmprobs = mnl.mnl_simulate(
                cmdata, cmbeta, numclasses, GPU=GPU, returnprobs=1)

            csprobs = []
            for cno in range(numclasses):
                tmp = mnl.mnl_simulate(
                    csdata, csbeta[cno], numalts, GPU=GPU, returnprobs=1)
                tmp = np.sum(tmp * chosen, axis=1)  # keep only chosen probs
                csprobs.append(np.reshape(tmp, (-1, 1)))
            csprobs = np.concatenate(csprobs, axis=1)

            h = csprobs * cmprobs
            loglik = np.sum(np.log(np.sum(h, axis=1)))
            wts = h / np.reshape(np.sum(h, axis=1), (-1, 1))
            return loglik, wts

        oldloglik = loglik
        loglik, wts = expectation(cmbeta, csbeta)
        if l_0 is None:
            l_0 = loglik
        print "current cmbeta", cmbeta
        print "current csbeta", csbeta
        print "current loglik", loglik, i + 1, "\n\n"
        if abs(loglik - oldloglik) < emtol:
            break

        # MAXIMIZATION

        for cno in range(numclasses):
            print "Estimating class specific model for class %d" % (cno + 1)
            t1 = time.time()
            weights = np.reshape(wts[:, cno], (-1, 1))
            fit, results = mnl.mnl_estimate(
                csdata, chosen, numalts, GPU=GPU, weights=weights,
                beta=csbeta[cno])
            print "Finished in %fs" % (time.time() - t1)
            csbeta[cno] = zip(*results)[0]
            results_d['cs%d' % cno] = results

        print "Estimating class membership model"
        t1 = time.time()
        fit, results = mnl.mnl_estimate(
            cmdata, None, numclasses, GPU=GPU, weights=wts, lcgrad=True,
            beta=cmbeta, coeffrange=(-1000, 1000))
        print "Finished in %fs" % (time.time() - t1)
        cmbeta = zip(*results)[0]
        results_d['cm'] = results

    l_1 = loglik
    l_0, foo = expectation(
        np.zeros(len(cmbeta)), [np.zeros(len(a)) for a in csbeta])
    ll_ratio = 1 - (l_1 / l_0)

    print "Null Log-liklihood: %f" % l_0
    print "Log-liklihood at convergence: %f" % l_1
    print "Log-liklihood ratio: %f" % ll_ratio

    a = []
    fnames = []
    fnames += cmfnames
    a += results_d['cm']
    for i in range(numclasses):
        fnames += ['%s cls%d' % (s, i) for s in csfnames]
        a += results_d['cs%d' % i]

    print misc.resultstotable(fnames, a)
    fit = (l_0, l_1, ll_ratio)
    misc.resultstocsv(
        fit, fnames, a, "lc-coeff.csv",
        tblname="Latent Class Model Coefficients")

    return (l_0, l_1, ll_ratio), results_d


def lcmnl_simulate(cmdata, numclasses, csdata, numalts, betas):
    cmfnames = ['cm%d' % i for i in range(cmdata.shape[1])]
    cmdata, newcmfnames = prep_cm_data(cmdata, numclasses, cmfnames)
    cmchoices = mnl.mnl_simulate(
        cmdata, betas['cm'], numclasses, GPU=GPU, returnprobs=0)
    cspdf = {}
    for cls in range(numclasses):
        cspdf['cs%d' % cls] = mnl.mnl_simulate(
            csdata, betas['cs%d' % cls], numalts, GPU=GPU, returnprobs=1
        ).flatten()
    cschoices = None
    return cmchoices, cschoices, cspdf

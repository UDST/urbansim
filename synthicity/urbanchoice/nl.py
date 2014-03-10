import time

import numpy as np
import scipy.optimize

import pmat
from pmat import PMAT

DEBUG = 1


class NestInfo:
    # nests is a dict with keys of nest ids and values of counts per nest
    # samplepernest = num alts sampled per nest, e.g. 100
    # chosennest is an array of length num observations which contains the
    # nest of the chosen alt

    def __init__(self, nests, samplepernest, chosennest, nestcounts=None):
        self._nests = nests
        self._samplepernest = samplepernest
        self._chosennest = chosennest
        self._ratepanel = None
        self._mcfaddencorrectionvec = None
        if nestcounts is None:
            self._nestcounts = nests.values
        else:
            self._nestcounts = nestcounts

    def numnests(self):
        return len(self._nests)

    def nestids(self):
        return self._nests.keys()

    def totaltspernest(self):
        return self._nestcounts

    def chosennest(self):
        return self._chosennest

    def samplepernest(self):
        return self._samplepernest

    def nestsizevaries(self):
        return self.totaltspernest().ndim != 1

    def mcfaddencorrectionvec(self, atype):
        if self._mcfaddencorrectionvec:
            return self._mcfaddencorrectionvec

        totaltspernest = self.totaltspernest()
        if not self.nestsizevaries():
            mcfaddencorrection = totaltspernest / float(self._samplepernest)
            mcfaddencorrectionvec = PMAT(
                np.reshape(
                    np.repeat(mcfaddencorrection, self._samplepernest),
                    (-1, 1)),
                atype).log(inplace=True)
        else:
            # can choose between taking the mean nest size or actual nest size
            # which varies per choice in this case
            # mcfaddencorrection = np.mean(
            #     totaltspernest,axis=0)/float(self._samplepernest)
            mcfaddencorrection = np.repeat(
                totaltspernest / float(self._samplepernest),
                self._samplepernest)
            mcfaddencorrection = np.transpose(
                np.reshape(
                    mcfaddencorrection,
                    (-1, self.samplepernest() * self.numnests())))
            mcfaddencorrectionvec = PMAT(
                mcfaddencorrection, atype).log(inplace=True)

        self._mcfaddencorrectionvec = mcfaddencorrectionvec
        return self._mcfaddencorrectionvec

    def ratepanel(self, atype):
        if self._ratepanel:
            return self._ratepanel

        numnests = self.numnests()
        nestsize = self.samplepernest()
        totaltspernest = self.totaltspernest()
        chosennest = self.chosennest()

        if totaltspernest.ndim == 1:
            rate_notchosen_panel = np.tile(
                totaltspernest / float(nestsize), (chosennest.size, 1))
            rate_chosen_panel = np.tile(
                (totaltspernest - 1) / float(nestsize - 1),
                (chosennest.size, 1))
        else:
            # in these case, the number of alternatives varies by decision
            rate_notchosen_panel = totaltspernest / float(nestsize)
            rate_chosen_panel = (totaltspernest - 1) / float(nestsize - 1)
            # in nest two lines, if there are a fewer alts than sample size,
            # they count fully
            # this is because of availability, which adds no utility for alts
            # that aren't available
            rate_notchosen_panel[np.where(rate_notchosen_panel < 1.0)] = 1.0
            rate_chosen_panel[np.where(rate_chosen_panel < 1.0)] = 1.0

        chosen = np.zeros((chosennest.size, numnests), dtype='bool')
        chosen[np.arange(chosennest.size), chosennest] = True

        rate_panel = rate_chosen_panel * chosen + \
            rate_notchosen_panel * np.invert(chosen)
        rate_panel = np.repeat(rate_panel, nestsize, axis=1)
        rate_panel[np.arange(chosennest.size), chosennest * nestsize] = 1

        self._ratepanel = PMAT(np.transpose(rate_panel), atype)
        return self._ratepanel

# right now MNL can only estimate location choice models, where every equation
# is the same
# it might be better to use stats models for a non-location choice problem

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS


def nl_probs(data, beta, mu, numalts, nestinfo, availability, GPU=0):

    atype = 'numpy' if not GPU else 'cuda'
    nestsize = nestinfo.samplepernest()

    utilities = beta.multiply(data)
    utilities.reshape(numalts, utilities.size() / numalts)

    if DEBUG:
        print "beta", beta, "mu", mu

    rate_panel = nestinfo.ratepanel(atype)
    assert rate_panel.shape() == utilities.shape()

    muvec = PMAT(np.reshape(np.repeat(mu, nestsize), (-1, 1)), atype)

    exponentiated_utility = utilities.multiply_by_col(
        muvec, inplace=False
    ).exp(inplace=True).element_multiply(rate_panel, inplace=True)

    if availability is not None:
        exponentiated_utility.element_multiply(availability, inplace=True)
    exponentiated_utility.reshape(nestsize, -1)

    sum_exponentiated_utility = exponentiated_utility.sum(
        axis=0).reshape(mu.size, -1)

    logGnest = sum_exponentiated_utility.log(inplace=True) \
        .multiply_by_col(PMAT(np.reshape(1.0 / mu - 1.0, (-1, 1)), atype))

    muvec = PMAT(np.reshape(np.repeat(mu - 1.0, nestsize), (-1, 1)), atype)

    logG = (utilities.multiply_by_col(muvec, inplace=False)
            .reshape(nestsize, -1)
            .add_row_vec(logGnest.reshape(1, -1), inplace=True)
            .reshape(numalts, -1))

    if not nestinfo.nestsizevaries():
        exponentiated_utility = \
            (utilities.element_add(logG, inplace=True)
             .add_col_vec(nestinfo.mcfaddencorrectionvec(atype), inplace=True)
             .exp(inplace=True))
    else:
        exponentiated_utility = \
            (utilities.element_add(logG, inplace=True)
             .element_add(nestinfo.mcfaddencorrectionvec(atype), inplace=True)
             .exp(inplace=True))

    if availability is not None:
        exponentiated_utility.element_multiply(availability, inplace=True)

    sum_exponentiated_utility = exponentiated_utility.sum(axis=0)

    probs = exponentiated_utility.divide_by_row(
        sum_exponentiated_utility, inplace=True)
    return probs


def get_hessian(derivative):
    return np.linalg.inv(np.dot(derivative, np.transpose(derivative)))


def get_standard_error(hessian):
    return np.sqrt(np.diagonal(hessian))


# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS
def nl_loglik(beta, data, chosen, numalts, nestinfo, availability,
              GPU=False, stderr=0):

    numvars = beta.size - nestinfo.numnests()
    numobs = data.size() / numvars / numalts

    mu, beta = beta[:nestinfo.numnests()], beta[nestinfo.numnests():]

    beta = np.reshape(beta, (1, beta.size))
    beta = PMAT(beta, data.typ)

    probs = nl_probs(data, beta, mu, numalts, nestinfo, availability, GPU)

    if stderr:
        assert 0  # return get_standard_error(get_hessian(gradmat.get_mat()))

    loglik = probs.element_multiply(chosen, inplace=True).sum(
        axis=0).log(inplace=True).sum(axis=1)

    if loglik.typ == 'numpy':
        loglik = loglik.get_mat()
    else:
        loglik = loglik.get_mat()[0, 0]

    if DEBUG:
        print "loglik", loglik
    return -1 * loglik

# numalts is now all the alts for all nests


def nl_estimate(data, chosen, numalts, nestinfo, availability,
                GPU=False, coeffrange=(-2.0, 2.0)):
    atype = 'numpy' if not GPU else 'cuda'

    data = np.transpose(data)
    chosen = np.transpose(chosen)
    if availability is not None:
        availability = np.transpose(availability)

    numvars = data.shape[0]
    numobs = data.shape[1] / numalts

    data, chosen = PMAT(data, atype), PMAT(chosen, atype)
    if availability is not None:
        availability = PMAT(availability, atype)

    beta = np.ones(nestinfo.numnests() + numvars)
    beta[:nestinfo.numnests()] = 4.0
    bounds = np.array(
        [coeffrange for i in range(nestinfo.numnests() + numvars)])
    bounds[:nestinfo.numnests()] = (1.0, 5.0)
    print "WARNING: setting bounds manually"

    t1 = time.time()
    args = (data, chosen, numalts, nestinfo, availability, GPU)
    bfgs_result = scipy.optimize.fmin_l_bfgs_b(nl_loglik,
                                               beta,
                                               args=args,
                                               approx_grad=True,
                                               bounds=bounds,
                                               epsilon=.001, pgtol=.01
                                               )
    # bfgs_result = scipy.optimize.fmin_bfgs(nl_loglik,
    #                                beta,
    #                                full_output=1,
    #                                args=(data,chosen,numalts,nestinfo,GPU))
    print "Optimized in %f seconds" % (time.time() - t1)

    beta = bfgs_result[0]
    inv_hessian = 1.0 / \
        approximate_second_derivative(nl_loglik, beta, args=args)
    stderr = np.sqrt(inv_hessian)  # get_standard_error(inv_hessian)
    tscore = beta / stderr

    l_0beta = np.zeros(nestinfo.numnests() + numvars)
    l_0beta[:nestinfo.numnests()] = 1.0
    l_0 = -1 * nl_loglik(l_0beta, *args)
    l_1 = -1 * nl_loglik(beta, *args)

    ll_ratio = 1 - (l_1 / l_0)
    print "Null Log-liklihood: %f" % l_0
    print "Log-liklihood at convergence: %f" % l_1
    print "Log-liklihood ratio: %f" % ll_ratio

    return (l_0, l_1, ll_ratio), zip(beta, stderr, tscore)


def approximate_second_derivative(f, x, args):
    delta = 0.001 * x
    mu = np.identity(delta.size, dtype='bool8')
    result = np.zeros(delta.size)
    delta_square = delta * delta
    for i in range(result.size):
        result[i] = ((f(x + 2 * delta[i] * mu[:, i], *args) - 2 *
                      f(x + delta[i] * mu[:, i], *args) + f(x, *args)) /
                     delta_square[i])
    return result

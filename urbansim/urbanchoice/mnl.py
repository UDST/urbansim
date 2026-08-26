"""
Number crunching code for multinomial logit.
``mnl_estimate`` and ``mnl_simulate`` especially are used by
``urbansim.models.lcm``.

"""
import logging

import numpy as np
import pandas as pd
import scipy.optimize

from ..utils.logutil import log_start_finish

logger = logging.getLogger(__name__)

# right now MNL can only estimate location choice models, where every equation
# is the same
# it might be better to use stats models for a non-location choice problem

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS


def mnl_probs(data, beta, numalts):
    logging.debug('start: calculate MNL probabilities')
    if numalts == 0:
        raise Exception("Number of alternatives is zero")
    utilities = np.dot(beta, data)
    utilities = np.reshape(utilities, (numalts, utilities.size // numalts),
                          order='F')

    # https://stats.stackexchange.com/questions/304758/softmax-overflow
    utilities = utilities - utilities.max(0, keepdims=True)

    exponentiated_utility = np.exp(utilities)
    exponentiated_utility[np.isinf(exponentiated_utility)] = 1e20
    np.clip(exponentiated_utility, 1e-300, None, out=exponentiated_utility)
    sum_exponentiated_utility = exponentiated_utility.sum(axis=0, keepdims=True)
    probs = exponentiated_utility / sum_exponentiated_utility
    probs[np.isnan(probs)] = 1e-300
    np.clip(probs, 1e-300, None, out=probs)

    logging.debug('finish: calculate MNL probabilities')
    return probs


def get_hessian(derivative):
    return np.linalg.inv(np.dot(derivative, np.transpose(derivative)))


def get_standard_error(hessian):
    return np.sqrt(np.diagonal(hessian))

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS


def mnl_loglik(beta, data, chosen, numalts, weights=None, lcgrad=False,
               stderr=0):
    logger.debug('start: calculate MNL log-likelihood')
    numvars = beta.size
    numobs = data.size // numvars // numalts

    beta = np.reshape(beta, (1, beta.size))

    probs = mnl_probs(data, beta, numalts)

    # lcgrad is the special gradient for the latent class membership model
    if lcgrad:
        assert weights
        gradmat = np.reshape(weights - probs, (probs.size, 1), order='F')
        gradarr = np.dot(data, gradmat)
    else:
        if not weights:
            gradmat = np.reshape(chosen - probs, (probs.size, 1), order='F')
        else:
            gradmat = np.reshape((chosen - probs) * weights,
                                 (probs.size, 1), order='F')
        gradarr = np.dot(data, gradmat)

    if stderr:
        gradmat = data * np.reshape(gradmat, (1, gradmat.size), order='F')
        gradmat = np.reshape(gradmat, (numvars, numalts * numobs), order='F')
        return get_standard_error(get_hessian(gradmat))

    chosen = np.reshape(chosen, (numalts, numobs), order='F')
    if weights is not None:
        if probs.shape == weights.shape:
            loglik = ((np.log(probs) * weights * chosen)
                      .sum(axis=1, keepdims=True).sum(axis=0, keepdims=True))
        else:
            loglik = ((np.log(probs) * (weights * chosen))
                      .sum(axis=1, keepdims=True).sum(axis=0, keepdims=True))
    else:
        loglik = ((np.log(probs) * chosen)
                  .sum(axis=1, keepdims=True).sum(axis=0, keepdims=True))

    gradarr = gradarr.flatten()

    logger.debug('finish: calculate MNL log-likelihood')
    return -1 * loglik, -1 * gradarr


def mnl_simulate(data, coeff, numalts, returnprobs=True):
    """
    Get the probabilities for each chooser choosing between `numalts`
    alternatives.

    Parameters
    ----------
    data : 2D array
        The data are expected to be in "long" form where each row is for
        one alternative. Alternatives are in groups of `numalts` rows per
        choosers. Alternatives must be in the same order for each chooser.
    coeff : 1D array
        The model coefficients corresponding to each column in `data`.
    numalts : int
        The number of alternatives available to each chooser.
    returnprobs : bool, optional
        If True, return the probabilities for each chooser/alternative instead
        of actual choices.

    Returns
    -------
    probs or choices: 2D array
        If `returnprobs` is True the probabilities are a 2D array with a
        row for each chooser and columns for each alternative.

    """
    logger.debug(
        'start: MNL simulation with len(data)={} and numalts={}'.format(
            len(data), numalts))

    data = np.transpose(data)
    coeff = np.reshape(np.array(coeff), (1, len(coeff)))

    probs = mnl_probs(data, coeff, numalts)

    if returnprobs:
        return np.transpose(probs)

    probs = np.cumsum(probs, axis=0)
    r = np.random.uniform(size=probs.size // numalts).reshape(1, -1)
    choices = (probs - r).argmax(axis=0)

    logger.debug('finish: MNL simulation')
    return choices


def mnl_estimate(data, chosen, numalts, coeffrange=(-3, 3),
                 weights=None, lcgrad=False, beta=None):
    """
    Calculate coefficients of the MNL model.

    Parameters
    ----------
    data : 2D array
        The data are expected to be in "long" form where each row is for
        one alternative. Alternatives are in groups of `numalts` rows per
        choosers. Alternatives must be in the same order for each chooser.
    chosen : 2D array
        This boolean array has a row for each chooser and a column for each
        alternative. The column ordering for alternatives is expected to be
        the same as their row ordering in the `data` array.
        A one (True) indicates which alternative each chooser has chosen.
    numalts : int
        The number of alternatives.
    coeffrange : tuple of floats, optional
        Limits of (min, max) to which coefficients are clipped.
    weights : ndarray, optional
    lcgrad : bool, optional
    beta : 1D array, optional
        Any initial guess for the coefficients.

    Returns
    -------
    log_likelihood : dict
        Dictionary of log-likelihood values describing the quality of the
        model fit.
    fit_parameters : pandas.DataFrame
        Table of fit parameters with columns 'Coefficient', 'Std. Error',
        'T-Score'. Each row corresponds to a column in `data` and are given
        in the same order as in `data`.

    See Also
    --------
    scipy.optimize.fmin_l_bfgs_b : The optimization routine used.

    """
    logger.debug(
        'start: MNL fit with len(data)={} and numalts={}'.format(
            len(data), numalts))

    numvars = data.shape[1]
    numobs = data.shape[0] // numalts

    if chosen is None:
        chosen = np.ones((numobs, numalts))  # used for latent classes

    data = np.transpose(data)
    chosen = np.transpose(chosen)

    if weights is not None:
        weights = np.transpose(weights)

    if beta is None:
        beta = np.zeros(numvars)
    bounds = [coeffrange] * numvars

    with log_start_finish('scipy optimization for MNL fit', logger):
        args = (data, chosen, numalts, weights, lcgrad)
        bfgs_result = scipy.optimize.fmin_l_bfgs_b(mnl_loglik,
                                                   beta,
                                                   args=args,
                                                   fprime=None,
                                                   factr=10,
                                                   approx_grad=False,
                                                   bounds=bounds
                                                   )

    if bfgs_result[2]['warnflag'] > 0:
        logger.warn("mnl did not converge correctly: %s",  bfgs_result)

    beta = bfgs_result[0]
    stderr = mnl_loglik(
        beta, data, chosen, numalts, weights, stderr=1, lcgrad=lcgrad)

    l0beta = np.zeros(numvars)
    l0 = -1 * mnl_loglik(l0beta, *args)[0]
    l1 = -1 * mnl_loglik(beta, *args)[0]

    log_likelihood = {
        'null': float(l0[0][0]),
        'convergence': float(l1[0][0]),
        'ratio': float((1 - (l1 / l0))[0][0])
    }

    fit_parameters = pd.DataFrame({
        'Coefficient': beta,
        'Std. Error': stderr,
        'T-Score': beta / stderr})

    logger.debug('finish: MNL fit')
    return log_likelihood, fit_parameters

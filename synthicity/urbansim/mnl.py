import scipy.optimize
import numpy as np
from pmat import PMAT
import pmat

# right now MNL can only estimate location choice models, where every equation is the same
# it might be better to use stats models for a non-location choice problem

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS
def mnl_probs(data,beta,numalts):

    utilities = beta.multiply(data)
    utilities.reshape(numalts,utilities.size()/numalts)

    exponentiated_utility = utilities.exp(inplace=True)
    sum_exponentiated_utility = exponentiated_utility.sum(axis=0)
    probs = exponentiated_utility.divide_by_row(sum_exponentiated_utility,inplace=True)

    return probs
    
def get_hessian(derivative):
    return np.linalg.inv(np.dot(derivative,np.transpose(derivative)))

def get_standard_error(hessian):
    return np.sqrt(np.diagonal(hessian))

# data should be column matrix of dimensions NUMVARS x (NUMALTS*NUMOBVS)
# beta is a row vector of dimensions 1 X NUMVARS
def mnl_loglik(beta,data,chosen,numalts,weights=None,lcgrad=False,stderr=0): 

    numvars = beta.size
    numobs = data.size()/numvars/numalts

    beta = np.reshape(beta,(1,beta.size))
    beta = PMAT(beta,data.typ)

    probs = mnl_probs(data,beta,numalts)
    
    if lcgrad: 
      assert weights
      gradmat = weights.subtract(probs).reshape(1,probs.size())
    else:
      gradmat = chosen.subtract(probs).reshape(1,probs.size())
    gradmat = data.multiply_by_row(gradmat)
    # this line is a bit hackish - you can't do the whole sum at once on a gpu
    # need to shorten the length of the axis over which to sum
    gradarr = gradmat.reshape(numvars*numalts,numobs)
    if weights is not None and not lcgrad: gradarr = gradarr.element_multiply(weights,inplace=True)
    gradarr = gradarr.sum(axis=1).reshape(numvars,numalts).sum(axis=1)

    gradmat.reshape(numvars,numalts*numobs)
    if stderr:
      if not lcgrad: return get_standard_error(get_hessian(gradmat.get_mat()))
      else: return np.zeros(beta.size())

    chosen.reshape(numalts,numobs)
    if weights is not None:
      loglik = (probs.log(inplace=True).element_multiply(weights,inplace=True) \
                                       .element_multiply(chosen,inplace=True)).sum(axis=1).sum(axis=0)
    else:
      loglik = (probs.log(inplace=True).element_multiply(chosen,inplace=True)).sum(axis=1).sum(axis=0)

    if loglik.typ == 'numpy':
        loglik, gradarr = loglik.get_mat(), gradarr.get_mat()
    else:
        loglik = loglik.get_mat()[0,0]
        gradarr = np.reshape(gradarr.get_mat(),(1,gradarr.size()))[0]

    return -1*loglik, -1*gradarr

def mnl_simulate(data, coeff, numalts, GPU=0, returnprobs=0):

    atype = 'numpy' if not GPU else 'cuda'

    data = np.transpose(data)
    coeff = np.reshape(np.array(coeff),(1,len(coeff)))

    data, coeff = PMAT(data,atype), PMAT(coeff,atype)

    probs = mnl_probs(data,coeff,numalts)
    
    if returnprobs: return np.transpose(probs.get_mat())

    # convert to cpu from here on - gpu doesn't currently support these ops
    if probs.typ == 'cuda': probs = PMAT(probs.get_mat()) 

    probs = probs.cumsum(axis=0)
    r = pmat.random(probs.size()/numalts)
    choices = probs.subtract(r,inplace=True).firstpositive(axis=0)

    return choices.get_mat()

def mnl_estimate(data,chosen,numalts,GPU=0,coeffrange=(-3,3),weights=None,lcgrad=False):

    atype = 'numpy' if not GPU else 'cuda'

    numvars = data.shape[1]
    numobs = data.shape[0]/numalts
    
    if chosen is None: chosen = np.ones((numobs,numalts)) # used for latent classes

    data = np.transpose(data)
    chosen = np.transpose(chosen)

    data, chosen = PMAT(data,atype), PMAT(chosen,atype)
    if weights is not None: weights = PMAT(np.transpose(weights),atype)

    beta = np.zeros(numvars)
    bounds = np.array([coeffrange for i in range(numvars)])
    
    args=(data,chosen,numalts,weights,lcgrad)
    bfgs_result = scipy.optimize.fmin_l_bfgs_b(mnl_loglik, 
                                    beta, 
                                    args=args,
                                    fprime=None,
                                    factr=1e9,
                                    approx_grad=False, 
                                    bounds=bounds
                                    )

    beta = bfgs_result[0]
    stderr = mnl_loglik(beta,data,chosen,numalts,weights,stderr=1,lcgrad=lcgrad) 
    tscore = beta / stderr

    l_0beta = np.zeros(numvars)
    l_0 = -1*mnl_loglik(l_0beta,*args)[0]
    l_1 = -1*mnl_loglik(beta,*args)[0]

    ll_ratio = 1-(l_1/l_0)
    print "Null Log-liklihood: %f" % l_0
    print "Log-liklihood at convergence: %f" % l_1
    print "Log-liklihood ratio: %f" % ll_ratio

    return (l_0,l_1,ll_ratio), zip(beta, stderr, tscore)

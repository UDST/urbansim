import random
import sys
import time

import numpy as np
import pandas as pd

import mnl
import nl
import pmat

GPU = 0


def enable_gpu():
    global GPU
    GPU = 1
    pmat.initialize_gpu()


def estimate(data, est_params, numalts, availability=None, gpu=None):
    global GPU
    if gpu is None:
        gpu = GPU
    if est_params[0] == 'mnl':
        if availability:
            assert 0  # not implemented yet
        return mnl.mnl_estimate(data, est_params[1], numalts, gpu)
    elif est_params[0] == 'nl':
        return nl.nl_estimate(
            data, est_params[1], numalts, est_params[2], availability, gpu)

    else:
        assert 0


def simulate(data, coeff, numalts, gpu=GPU):
    return mnl.mnl_simulate(data, coeff, numalts, gpu)

# if nested logit, add the field names for the additional nesting params


def add_fnames(fnames, est_params):
    if est_params[0] == 'nl':
        fnames = ['mu%d' % (i + 1)
                  for i in range(est_params[2].numnests())] + fnames
    return fnames


def mnl_estimate(data, chosen, numalts, gpu=GPU):
    return mnl.mnl_estimate(data, chosen, numalts, gpu)


def mnl_simulate(data, coeff, numalts, gpu=GPU, returnprobs=0):
    return mnl.mnl_simulate(data, coeff, numalts, gpu, returnprobs)


def mnl_interaction_dataset(choosers, alternatives, SAMPLE_SIZE,
                            chosenalts=None):
    if chosenalts is not None:
        isin = chosenalts.isin(alternatives.index)
        removing = isin.value_counts()[False]
        if removing:
            print (
                "Removing {} choice situations because chosen "
                "alt doesn't exist"
            ).format(removing)
            choosers = choosers[isin]
            chosenalts = chosenalts[isin]

    numchoosers = choosers.shape[0]
    numalts = alternatives.shape[0]

    if SAMPLE_SIZE < numalts:
        sample = np.random.choice(
            alternatives.index.values, SAMPLE_SIZE * choosers.shape[0])
        if chosenalts is not None:
            # replace with chosen alternative
            sample[::SAMPLE_SIZE] = chosenalts
    else:
        assert chosenalts is None  # if not sampling, must be simulating
        # we're about to do a huge join - do this with a discretized population
        assert numchoosers < 10
        sample = np.tile(alternatives.index.values, numchoosers)

    if not choosers.index.is_unique:
        raise Exception(
            "ERROR: choosers index is not unique, "
            "sample will not work correctly")
    if not alternatives.index.is_unique:
        raise Exception(
            "ERROR: alternatives index is not unique, "
            "sample will not work correctly")

    alts_sample = alternatives.loc[sample]
    assert len(alts_sample.index) == SAMPLE_SIZE * len(choosers.index)
    try:
        alts_sample['join_index'] = np.repeat(choosers.index, SAMPLE_SIZE)
    except:
        raise Exception(
            "ERROR: An exception here means agents and "
            "alternatives aren't merging correctly")

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='join_index', right_index=True,
        suffixes=('', '_r'))

    chosen = np.zeros((numchoosers, SAMPLE_SIZE))
    chosen[:, 0] = 1

    return sample, alts_sample, ('mnl', chosen)


def mnl_choice_from_sample(sample, choices, SAMPLE_SIZE):
    return np.reshape(
        sample, (choices.size, SAMPLE_SIZE))[np.arange(choices.size), choices]


def nl_estimate(data, chosen, numalts, nestinfo, gpu=GPU):
    return nl.nl_estimate(data, chosen, numalts, nestinfo, gpu)


def nl_simulate(data, coeff, numalts, gpu=GPU):
    return nl.nl_simulate(data, coeff, numalts, gpu)


def nl_interaction_dataset(choosers, alternatives, SAMPLE_SIZE, nestcol,
                           chosenalts=None, left_on='nodeid', presample=None,
                           nestcounts=None):
    nests = alternatives[nestcol].value_counts()
    print "Alternatives in each nest\n", nests
    sample_size_per_nest = SAMPLE_SIZE / len(nests)
    print "Sample size per nest", sample_size_per_nest
    assert SAMPLE_SIZE % len(nests) == 0  # divides evenly

    if presample is None:
        sample = None
        for m in np.sort(nests.keys().values):
            # full sampled set
            nsample = np.random.choice(
                alternatives[alternatives[nestcol] == m].index.values,
                sample_size_per_nest * choosers.shape[0])
            nsample = np.reshape(
                nsample, (nsample.size / sample_size_per_nest,
                          sample_size_per_nest))
            if sample is None:
                sample = nsample
            else:
                sample = np.concatenate((sample, nsample), axis=1)
    else:
        sample = presample

    # means we're estimating, not simulating
    if chosenalts is not None:
        chosen = np.zeros((choosers.shape[0], SAMPLE_SIZE))
        if isinstance(left_on, str):
            assert left_on in choosers.columns
        chosennest = pd.merge(
            choosers, alternatives, left_on=left_on, right_index=True,
            how="left"
        ).set_index(choosers.index)[nestcol]  # need to maintain same index
        print "Chosen alternatives by nest\n", chosennest.value_counts()
        assert sample.shape == chosen.shape
        # this restriction should be removed in the future - for now nestids
        # have to count from 0 to numnests-1
        assert min(nests.keys()) == 0 and max(nests.keys()) == len(nests) - 1
        # replace with chosen alternative
        sample[range(sample.shape[0]), chosennest *
               sample_size_per_nest] = chosenalts
        chosen[range(chosen.shape[0]), chosennest * sample_size_per_nest] = 1

    sample = sample.flatten().astype('object')
    alts_sample = alternatives.ix[sample]
    alts_sample['join_index'] = np.repeat(choosers.index, SAMPLE_SIZE)

    print "Merging sample (takes a while)"
    t1 = time.time()
    alts_sample = pd.merge(
        alts_sample, choosers, left_on='join_index', right_index=True)
    print "Done merging sample in %f" % (time.time() - t1)

    nestinfo = nl.NestInfo(nests, sample_size_per_nest, chosennest, nestcounts)

    return sample, alts_sample, ('nl', chosen, nestinfo)

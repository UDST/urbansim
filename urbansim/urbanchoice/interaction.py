"""
Utilities for making merged interaction tables of choosers and
the alternatives from which they are choosing.
Used for location choice models.

"""
import logging
import random
import sys
import time

import numpy as np
import pandas as pd

import mnl
import nl
import pmat

logger = logging.getLogger(__name__)
GPU = False


def enable_gpu():
    global GPU
    GPU = 1
    pmat.initialize_gpu()


# if nested logit, add the field names for the additional nesting params


def add_fnames(fnames, est_params):
    if est_params[0] == 'nl':
        fnames = ['mu%d' % (i + 1)
                  for i in range(est_params[2].numnests())] + fnames
    return fnames


# TODO: split this out into separate functions for estimation
# and simulation.
def mnl_interaction_dataset(choosers, alternatives, SAMPLE_SIZE,
                            chosenalts=None):
    logger.debug((
        'start: compute MNL interaction dataset with {} choosers, '
        '{} alternatives, and sample_size={}'
        ).format(len(choosers), len(alternatives), SAMPLE_SIZE))
    # filter choosers and their current choices if they point to
    # something that isn't in the alternatives table
    if chosenalts is not None:
        isin = chosenalts.isin(alternatives.index)
        try:
            removing = isin.value_counts().loc[False]
        except:
            removing = None
        if removing:
            logger.info((
                "Removing {} choice situations because chosen "
                "alt doesn't exist"
            ).format(removing))
            choosers = choosers[isin]
            chosenalts = chosenalts[isin]

    numchoosers = choosers.shape[0]
    numalts = alternatives.shape[0]

    # TODO: this is currently broken in a situation where
    # SAMPLE_SIZE >= numalts. That may not happen often in
    # practical situations but it should be supported
    # because a) why not? and b) testing.
    if SAMPLE_SIZE < numalts:
        sample = np.random.choice(
            alternatives.index.values, SAMPLE_SIZE * numchoosers)
        if chosenalts is not None:
            # replace the first row for each chooser with
            # the currently chosen alternative.
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
        # TODO: log the error here and re-raise the original exception
        raise Exception(
            "ERROR: An exception here means agents and "
            "alternatives aren't merging correctly")

    alts_sample = pd.merge(
        alts_sample, choosers, left_on='join_index', right_index=True,
        suffixes=('', '_r'))

    chosen = np.zeros((numchoosers, SAMPLE_SIZE))
    chosen[:, 0] = 1

    logger.debug('finish: compute MNL interaction dataset')
    return sample, alts_sample, chosen


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

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
import pmat

logger = logging.getLogger(__name__)
GPU = False


def enable_gpu():
    global GPU
    GPU = 1
    pmat.initialize_gpu()


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

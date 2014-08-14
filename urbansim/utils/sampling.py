import numpy as np
import pandas as pd

def sample_rows(total, sampling_data, with_replace=True, accounting_data=None, foreign_key_column=None, max_iterations=50):
	"""
	comments to go here...

    todo: make the accounting stuff either a delegate or a computed column??
	"""

	# simplest case, just return n random rows from the sampling frame
	if accounting_data is None:
		return sampling_data.loc[np.random.choice(sampling_data.index.values, total, replace=with_replace)].copy()

	# determine avg number of accounting records per sample (e.g. persons per household, jobs per business)
	# todo: throw an error if ther sampling row count is larger than the accounting row count
	per_sample = len(accounting_data.index) / (1.0 * len(sampling_data.index))
    
	# do the intial sample 
	num_samples = int(math.ceil(math.fabs(total) / per_sample))
    if with_replace:
    	sample_idx = sampling_data.index.values
        sample_ids = np.random.choice(sample_idx, num_samples)
    else:
    	sample_idx = np.random.permutation(sampling_data.index.values)
        sample_ids = sample_idx[0:num_samples]
        samplePos = num_samples

    sample_rows = sampling_data.loc[sample_ids].copy()
    curr_total = len(sample_rows.merge(accounting_data, left_index=True, right_on=foreign_key_column).index)

	# iteratively refine the sample until we match the accounting total
	for i in range(0, max_iterations)

		# keep going if we haven't hit the control
        remaining = total - curr_total
        if remaining == 0: break
        num_samples = int(math.ceil(math.fabs(remaining) / per_sample))

        if remaining > 0:
            # we're short, keep sampling 
            if with_replace:
            	curr_ids = np.random.choice(sample_idx, num_samples)
            else:
            	curr_ids = sample_idx[samplePos:samplePos + num_samples]
            	samplePos += num_samples
        else:
            # we've overshot, remove from existing samples (FIFO)
            # if replacement is false do these need to go back into the indexes?
            curr_rows = sample_rows[:numSamples]
            sample_rows = sampleRows[numSamples:]
            currTotal -= len(currRows.merge(child_data, left_index=True, right_on=key_field).index)

    return sample_rows.copy()
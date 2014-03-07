import os
import string
import sys

import simplejson

from synthicity.utils import misc
sys.path.insert(0, ".")
import dataset

args = sys.argv[1:]
dset = dataset.LocalDataset(args[0])
args = args[1:]

for arg in args:
    print "Generating %s" % arg
    item = string.split(arg, ',')
    if len(item) == 1:
        model, mode = item[0], "run"
    else:
        model, mode = item
    d = misc.run_model(model, dset, mode)
    # print d

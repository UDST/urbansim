import pandas as pd
import sys

args = sys.argv[1:]

assert len(args) == 1

store = pd.HDFStore(args[0])

print store

for key in store.keys():
    print "\n\nTABLENAME", key
    print store[key]
    print store[key].dtypes
    print store[key].describe()

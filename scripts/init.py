import os, json, sys, time, pandas as pd, numpy as np
from synthicity.urbansimd import urbansimd
from synthicity.utils import misc
from synthicity.utils.misc import run_model
pd.set_option('precision',3)
pd.set_option('display.width',160)
try: import couchdb
except: pass

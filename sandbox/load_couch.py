import json
import os

import couchdb

couch = couchdb.Server()  # Assuming localhost:5984
# If your CouchDB server is running elsewhere, set it up like this:
#couch = couchdb.Server('http://example.com:5984/')

# select database
db = couch['bayarea']

fnames = os.listdir('.')
for fname in fnames:
    if not fname.endswith('.json'):
        continue
    print fname
    doc = json.loads(open(fname).read())
    doc['_id'] = fname
    db.save(doc)

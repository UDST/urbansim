import json, urllib2

headers = {}
headers['Content-Type'] = 'application/json'
jdata = {
"tables":['parcels'],
#"tables":['residential_rent'],
#"variable":"""subset['building_sqft'].groupby(level=0).transform('sum') / \
#                         (subset['shape_area'].groupby(level=0).transform('mean')*10.7639)""",
#"variable":"""subset[(subset['building_type_id'] < 4)*(subset['year_built'] < 1950)].groupby('zone_id')['building_sqft'].transform('sum')""",
#"variable":"""subset[_YEARS_].groupby(level=0).mean()""",
"variable": "subset['parcel_acres']",
#"bbox":[550000,4180000,550100,4180100],
"bbox":[1500000,1500000,1500500,1500500],
#"srid":3740,
"srid":2903,
#"minmaxonly":1,
#"sampleprop":.0003,
"bins":4,
"quantile":1,
}
print jdata
req = urllib2.Request("http://paris.urbansim.org:8763/", json.dumps(jdata), headers)
print urllib2.urlopen(req).read()

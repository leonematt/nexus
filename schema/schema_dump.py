import json
import sys


def dump_names(jobj):
    for k in jobj.keys():
        print(f"         {k}")
        kobj = jobj[k]
        if kobj['type'] == 'object':
            dump_names(kobj['properties'])

with open(sys.argv[1]) as f:
    jobj = json.load(f)

props = jobj['properties']
dump_names(props)

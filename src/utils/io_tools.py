import json, os

def read_json(p): 
    with open(p,'r') as f: return json.load(f)

def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,'w') as f: json.dump(obj, f, indent=2)
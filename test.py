import json

with open('serviceAccountKey.json', 'r') as f:
    creds = json.load(f)
    
# This gives you a single-line JSON string
print(json.dumps(creds))
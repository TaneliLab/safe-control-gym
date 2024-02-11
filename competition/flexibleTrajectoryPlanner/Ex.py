import yaml
import os
print(os.getcwd())
filepath = os.path.join('..', 'planner.yaml')

with open(filepath, 'r') as file:
    data = yaml.safe_load(file)
print(data['globalplan'])

for d in data['globalplan']:
    print(d)
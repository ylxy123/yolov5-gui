import os
import json
import shutil
import yaml

with open("../data/hyps/hyp.scratch.yaml", "r", encoding='utf-8') as f:
    content = f.read()

yaml_cont = yaml.load(content, Loader=yaml.FullLoader)
print(yaml_cont)






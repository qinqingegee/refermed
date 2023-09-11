import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
import numpy as np
import torch.nn.functional as F
import transformers
import pandas as pd
import random
import copy
from PIL import Image
import tqdm
import os
# train读出来
# {img_id : 1, question:[], answer:[], 
# 把别的扔进去
 #"q_lang": "en", "location": "Abdomen", "modality": "MRI", "answer_type": "OPEN", "base_type": "vqa", "content_type": "Position", "triple": ["vhead", "_", "_"],}

#{   1:   {"question":[],    "answer":[],   "location": "Abdomen",    "modality": "MRI"    }}

DATA_DIR = '/home/yangjinxia/mine/dataset/Slake/Slake1.0'
with open(os.path.join(DATA_DIR, "{}.json".format("train"))) as f:
    data = json.load(f)
# print(data)
new = []
tmp = {}
for i in range(len(data)):
    if data[i]["q_lang"] != "en":
        continue
    id = int(data[i]["img_id"])
    if id not in tmp.keys():
        tmp[id]  = {}
    tmp[id]["img_name"] = data[i]["img_name"]
    if "question" not in tmp[id].keys():
        tmp[id]["question"] = [data[i]["question"]]
    else:
        tmp[id]["question"].append(data[i]["question"])
    if "answer" not in tmp[id].keys():
        tmp[id]["answer"] = [data[i]["answer"]]
    else:
        tmp[id]["answer"].append(data[i]["answer"])
    
    tmp[id]["location"] = data[i]["location"]
    tmp[id]["modality"] = data[i]["modality"]
    tmp[id]["answer_type"] = data[i]["answer_type"]
    tmp[id]["base_type"] = data[i]["base_type"]
    tmp[id]["content_type"] = data[i]["content_type"]
    tmp[id]["triple"] = data[i]["triple"]



with open("./slake_train.json", "w", encoding="utf-8") as f:
    json.dump(tmp, f, indent=4, ensure_ascii=False)


    





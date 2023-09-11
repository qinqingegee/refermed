import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pickle, json
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

#{ 1:  { "question":[],    "answer":[]    }     }
DATA_DIR = '/home/yangjinxia/mine/dataset/pvqa/qas'
with open(os.path.join(DATA_DIR, "{}/{}_qa.pkl".format("train","train")), "rb") as f:
    data = pickle.load(f)
# print(data)
new = []
tmp = {}
for i in range(len(data)):
    id = data[i]["image"]
    if id not in tmp.keys():
        tmp[id]  = {}
    tmp[id]["image"] = data[i]["image"]
    if "question" not in tmp[id].keys():
        tmp[id]["question"] = [data[i]["question"]]
    else:
        tmp[id]["question"].append(data[i]["question"])
    if "answer" not in tmp[id].keys():
        tmp[id]["answer"] = [data[i]["answer"]]
    else:
        tmp[id]["answer"].append(data[i]["answer"])

with open(DATA_DIR + "/train/pvqa_train.json", "w", encoding="utf-8") as f:
    json.dump(tmp, f, indent=4, ensure_ascii=False)


    





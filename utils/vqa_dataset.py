import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import pandas as pd
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    CAPTION_PROMPT,
)

def init_VQA_RAD(base_image_dir): # 返回json文件内容 包含了图像标题
    DATA_DIR = os.path.join(base_image_dir, "VQA-RAD")
    with open(os.path.join(DATA_DIR, "{}.json".format("train"))) as f:
        vqa_data = json.load(f)
    # return vqa_data["image_name"], vqa_data["question"], vqa_data["answer"]
    return vqa_data

def init_PMC_VQA(base_image_dir):
    DATA_DIR = os.path.join(base_image_dir, "PMC-VQA")
    data = pd.read_csv(os.path.join(DATA_DIR, "train_2.csv"))
    return data

def init_SLAKE(base_image_dir):
    DATA_DIR = os.path.join(base_image_dir, "Slake/Slake1.0")
    with open(os.path.join(DATA_DIR, "{}.json".format("slake_train"))) as f:
        slake_data = json.load(f)
    return slake_data

def init_Path_VQA(base_image_dir):
    DATA_DIR = os.path.join(base_image_dir, "pvqa/qas")
    with open(os.path.join(DATA_DIR, "{}/pvqa_{}.json".format("train", "train"))) as f:
        data = json.load(f)
    return data

class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="VQA_RAD||PMC_VQA||SLAKE||Path_VQA",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.caption_prompt = CAPTION_PROMPT

        # DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        # self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        # with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
        #     vqa_data = json.load(f)
        # vqa_data["image_name"], vqa_data["question"], vqa_data["answer"]
        self.vqa_data = vqa_data.split("||")
        
        self.data2list = {}
        for ds in self.vqa_data:
            data = eval("init_{}".format(ds))(base_image_dir) # eval用的好 直接执行init_ds函数
            self.data2list[ds] = data
            # "ade20k":(图像，mask)
            # self.data2classes[ds] = classes       
            # "ade20k": np.array【所有类别wall】

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.vqa_data) - 1)
        ds = self.vqa_data[ds]
        if ds in ["VQA_RAD"]:
            item = self.data2list[ds]
            # print("vqa_rad: ", len(item))
            idx = random.randint(0, len(item) - 1)
            # item = self.vqa_data[idx]
            pth = os.path.join(self.base_image_dir, "VQA-RAD/images")
            item = item[list(item.keys())[idx]]
            image_path = os.path.join(pth, item["image"])
            img = cv2.imread(image_path)
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images.shape[:2] # 除去了3 channel   640 427 
            images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]  # preprocess images for clip
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam
            resize = images.shape[:2] # 1024 683
           
            #     "from": "human",
            #     "value": "<image>\n What is the main color of the plate in the image?"
            #   },
            #   {
            #     "from": "gpt",

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # {'human': 'USER', 'gpt': 'ASSISTANT'}
            # N组对话
            conversations = []
            conv.messages = []
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + " " + item["question"][0] )
            conv.append_message(conv.roles[1], item["answer"][0])
            conversations.append(conv.get_prompt()) 

            for i in range(1, min(3, len(item["question"]))):
                conv.messages = []
                conv.append_message(conv.roles[0], item["question"][i] )
                conv.append_message(conv.roles[1], item["answer"][i])
                conversations.append(conv.get_prompt()) 
                 
        elif ds in ["PMC_VQA"]:
            item = self.data2list[ds]
            idx = random.randint(0, len(item["index"]) - 1)
            item = self.data2list[ds].iloc[idx]
            pth = os.path.join(self.base_image_dir, "PMC-VQA/images_2/figures")
            image_path = os.path.join(pth, item["Figure_path"])
            img = cv2.imread(image_path)
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images.shape[:2] # 除去了3 channel   640 427 
            images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]  # preprocess images for clip
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam
            resize = images.shape[:2] # 1024 683

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # {'human': 'USER', 'gpt': 'ASSISTANT'}
            # 2组对话
            conversations = []
            conv.messages = []

            Choice_A = item['Choice A'] 
            Choice_B = item['Choice B'] 
            Choice_C = item['Choice C'] 
            Choice_D = item['Choice D'] 
            Answer = item["Answer"].replace('A:','').replace('B:','').replace('C:','').replace('D:','')
            combined_choice = Choice_A +','+ Choice_B +','+ Choice_C + ','+ Choice_D
            # 两组对话
            conv.append_message(conv.roles[0], random.choice(self.caption_prompt))
            conv.append_message(conv.roles[1], item["Caption"])
            conversations.append(conv.get_prompt())
            conv.messages = []

            conv.append_message(conv.roles[0], item["Question"]+ ' The options are: ' + combined_choice)
            conv.append_message(conv.roles[1], Answer)
            conversations.append(conv.get_prompt()) 
        
        elif ds in ["SLAKE"]:
            item = self.data2list[ds]
            idx = random.randint(0, len(item) - 1)
            item = self.data2list[ds][list(self.data2list[ds].keys())[idx]]
            pth = os.path.join(self.base_image_dir, "Slake/Slake1.0/imgs")
            image_path = os.path.join(pth, item["img_name"])
        
            img = cv2.imread(image_path)
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images.shape[:2] # 除去了3 channel   640 427 
            images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]  # preprocess images for clip
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam
            resize = images.shape[:2] # 1024 683

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # {'human': 'USER', 'gpt': 'ASSISTANT'}
            # n组对话
            conversations = []
            conv.messages = []
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + " " + item["question"][0] )
            conv.append_message(conv.roles[1], item["answer"][0])
            conversations.append(conv.get_prompt()) 

            # for i in range(1, len(item["question"])):
            for i in range(1, 3):
                conv.messages = []
                conv.append_message(conv.roles[0], item["question"][i] )
                conv.append_message(conv.roles[1], item["answer"][i])
                conversations.append(conv.get_prompt()) 

        elif ds in ["Path_VQA"]:
            item = self.data2list[ds]
            idx = random.randint(0, len(list(item.keys())) - 1)
            item = self.data2list[ds][list(item.keys())[idx]]
            pth = os.path.join(self.base_image_dir, "pvqa/images/train")
            image_path = os.path.join(pth, "{}.jpg".format(item["image"]))
            img = cv2.imread(image_path)
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ori_size = images.shape[:2] # 除去了3 channel 
            images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
                "pixel_values"
            ][0]  # preprocess images for clip
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam
            resize = images.shape[:2] # 1024 683

            conv = get_default_conv_template(
                "vicuna"
            ).copy()  # conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # {'human': 'USER', 'gpt': 'ASSISTANT'}
            # n组对话
            conversations = []
            conv.messages = []
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + " " + item["question"][0] )
            conv.append_message(conv.roles[1], item["answer"][0])
            conversations.append(conv.get_prompt()) 
            for i in range(1, 3):
                conv.messages = []
                conv.append_message(conv.roles[0], item["question"][i] )
                conv.append_message(conv.roles[1], item["answer"][i])
                conversations.append(conv.get_prompt()) 
        questions = conversations
        sampled_classes = conversations

        # replace <image> token
        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len # 256
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size) # vqa 不要mask 0 640 427
        label = torch.ones(ori_size) * self.ignore_label # 全部是255

        return (
            image_path,
            images,
            images_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )

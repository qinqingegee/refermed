import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
from einops import repeat

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .utils import (
    ANSWER_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    SHORT_QUESTION_LIST,
)

def init_synapse(base_image_dir): # 这数据集有点问题：原图不变，mask一直在变化

    synapse_data_root = os.path.join(base_image_dir, "segment/synapse")
    sample_list = open(os.path.join(synapse_data_root, 'train.txt')).readlines()
    # classes = {1: "spleen",
    #             2: "right kidney",
    #             3: "left kidney",
    #             4: "gallbladder",
    #             5: "liver",
    #             6: "stomach",
    #             7: "aorta",
    #             8: "pancreas"}

    classes = [" ", "spleen", "right kidney", "left kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"] # 处理错了，第一个不应该是空，而是整体-1对到0索引
    return synapse_data_root, sample_list, classes

def init_kvasir(base_image_dir):
    data_root = os.path.join(base_image_dir, "segment/kvasir/Kvasir-SEG")
    classes = ["", "polyp"]

    train_data = glob.glob(
        os.path.join(data_root, "images", "*.jpg")
    )
    mask_data = [
        x.replace("images", "masks")
        for x in train_data
    ]
    return data_root, (train_data, mask_data), classes


def init_siim(base_image_dir):
    data_root = os.path.join(base_image_dir, "segment/siim")
    classes = ["","pneumothorax"]

    train_data = glob.glob(
        os.path.join(data_root, "images", "*.jpg")
    )
    mask_data = [
        x.replace("images", "masks")
        for x in train_data
    ]
    return data_root, (train_data, mask_data), classes
 

def init_rsna(base_image_dir):
    data_root = os.path.join(base_image_dir, "segment/rsna")
    classes = ["", "pneumonia"]

    train_data = glob.glob(
        os.path.join(data_root, "images", "*.jpg")
    )
    mask_data = [
        x.replace("images", "masks")
        for x in train_data
    ]
    return data_root, (train_data, mask_data), classes
 

 
class SemSegDataset(torch.utils.data.Dataset):
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
        sem_seg_data="synapse"  #||",synapse||siim||rsna||kvasir  
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

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.data2path = {}
        self.data2list = {}
        self.data2classes = {}
        # ade20k, cocostuff
        self.sem_seg_datas = sem_seg_data.split("||")
        for ds in self.sem_seg_datas:
            self.data2path[ds], self.data2list[ds], self.data2classes[ds] = eval("init_{}".format(ds))(base_image_dir) # eval用的好 直接执行init_ds函数
            # self.data2list[ds] = (images, labels) # "ade20k":(图像，mask)
            # self.data2classes[ds] = classes       # "ade20k": np.array【所有类别 wall 】
            
        if "cocostuff" in self.sem_seg_datas:
            self.cocostuff_class2index = {
                c: i for i, c in enumerate(self.data2classes["cocostuff"])
            }

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
        ds = random.randint(0, len(self.sem_seg_datas) - 1)
        ds = self.sem_seg_datas[ds]
        if ds in ["synapse"]:
            idx = random.randint(0, len(self.data2list[ds]) - 1)
            slice_name = self.data2list[ds][idx].strip('\n')
            image_path = os.path.join(self.data2path[ds],'train_npz', slice_name+'.npz')
            data = np.load(image_path)
            images, label = data['image'], data['label']  # 2维 512*512 没有通道维度，repeat三次  image一直没变，label随着切片在变 whywhy
            # images = torch.from_numpy(images).unsqueeze(2)  # 512*512*1
            # images = np.expand_dims(images, axis=2) # 512*512*1

            images = torch.from_numpy(images).unsqueeze(2).numpy()
            print(images[0])
            images = repeat(images, 'h w c -> h w (repeat c)', repeat=3)  #.numpy()
            
            # preprocess images for clip 不知npz文件行不行 维度不对s
            images_clip = self.clip_image_processor.preprocess(   # 检查图像是否合法, 故不通过; 直接用1维, normalize为3维,又不通过
                images, return_tensors="pt", do_normalize= False
            )["pixel_values"][0] # 按1维输进去，后面又repeat*3，还是有问题   
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam 512*512  就变了尺寸
            resize = images.shape[:2]  # 512 512 

            unique_label = np.unique(label).tolist()
            unique_label = [int(i) for i in unique_label]

            # 留不留
            # if 255 in unique_label:
            #     unique_label.remove(255)
            # if len(unique_label) == 0:
            #     return self.__getitem__(0)

            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample: # 如果包含了多个类，就随机选3个
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes

        elif ds in ["kvasir"]: #  no label的情况
            images, labels = self.data2list[ds]
            idx = random.randint(0, len(images) - 1)
            # data2classes
            image_path = images[idx]
            label_path = labels[idx]
            img = cv2.imread(image_path) #  530*619*3
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 530*619*3
 
            label = cv2.imread(label_path)[:,:,0]/255
            label = label.round().astype(int)
            # label = np.round(label, 0)

            # preprocess images for clip 
            images_clip = self.clip_image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"][0]   # 3*224*224
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size 256

            images = self.transform.apply_image(images)  # preprocess images for sam 
            resize = images.shape[:2]   # 
            unique_label = np.unique(label).tolist()
            # 留不留  0 1的判断
            if 0 in unique_label:
                unique_label.remove(0)

            if len(unique_label) == 0:
                return self.__getitem__(0)
            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample: # 如果包含了多个类，就随机选3个
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes

        elif ds in ["siim", "rsna"]:
            images, labels = self.data2list[ds]
            idx = random.randint(0, len(images) - 1)
            # data2classes
            image_path = images[idx]
            label_path = labels[idx]
            img = cv2.imread(image_path) #  530*619*3
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 530*619*3
            label = Image.open(label_path)
            label = np.array(label) / 255
            label = label.round().astype(int) # binarize to 0 or 1 
            
            # preprocess images for clip 
            images_clip = self.clip_image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"][0]   # 3*224*224
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size 256

            images = self.transform.apply_image(images)  # 512 512 3
            resize = images.shape[:2]   # [512,512]

            unique_label = np.unique(label).tolist()
            # 去除未编码的符号
            if 0 in unique_label:
                unique_label.remove(0)

            # if len(unique_label) == 0:
            #     return self.__getitem__(0)
            classes = [self.data2classes[ds][class_id] for class_id in unique_label]
            if len(classes) >= self.num_classes_per_sample: # 如果包含了多个类，就随机选3个
                sampled_classes = np.random.choice(
                    classes, size=self.num_classes_per_sample, replace=False
                ).tolist()
            else:
                sampled_classes = classes
                
            
        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes: # 随机选的类别
            text = sampled_cls
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list) # 随机
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))
            # if ds in ["kvasir"]:
            #     continue
            # 从文本映射为数字
            class_id = self.data2classes[ds].index(sampled_cls) # 找到在整个class列表中的id
            class_ids.append(class_id)

        # 构造对话
        conversations = []
        conv = get_default_conv_template("vicuna").copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1
        
        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        
        
        # if ds in ["synapse"]:
        #     images = torch.from_numpy(images.astype(np.float32)).unsqueeze(2)
        #     images = repeat(images, 'h w c -> h w (repeat c)', repeat=3).numpy()
        # （H, W, C）-> (C, H, W)
        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous()) # 3*1024*1024
        label = torch.from_numpy(label).long() # 1024 1024
       
        # if ds not in ["kvasir"]:
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0) # 不同类别对应的mask stack在一起
        # else:
        #     masks = label
        return (
            image_path,
            images,   # 3*1024*1024
            images_clip,  # 3*224*224
            conversations,
            masks,   # 1*1024*1024
            label,   # 1024*1024
            resize,  # 512 512
            questions,
            sampled_classes,
        )

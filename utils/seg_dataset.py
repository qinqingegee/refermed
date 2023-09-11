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

def init_synapse(base_image_dir):

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

    classes = [" ", "spleen", "right kidney", "left kidney", "gallbladder", "liver", "stomach", "aorta", "pancreas"]
    return synapse_data_root, sample_list, classes

def init_kvasir(base_image_dir):
    synapse_data_root = os.path.join(base_image_dir, "segment/kvasir/Kvasir-SEG")
    classes = ["polyp"]

    train_data = glob.glob(
        os.path.join(synapse_data_root, "images", "*.jpg")
    )
    mask_data = [
        x.replace("images", "masks")
        for x in train_data
    ]
    return synapse_data_root, (train_data, mask_data), classes
 

 
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
        sem_seg_data="kvasir"#||",synapse  
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
            images, label = data['image'], data['label']

            # preprocess images for clip 不知npz文件行不行
            images_clip = self.clip_image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"][0]
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam
            resize = images.shape[:2]

            unique_label = np.unique(label).tolist()

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

        elif ds in ["kvasir"]:
            images, labels = self.data2list[ds]
            idx = random.randint(0, len(images) - 1)
            # data2classes
            image_path = images[idx]
            label_path = labels[idx]
            img = cv2.imread(image_path)
            images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            label = Image.open(label_path) # mask
            label = np.array(label)
            # preprocess images for clip 
            images_clip = self.clip_image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"][0]
            image_token_len = (images_clip.shape[1] // 14) * (
                images_clip.shape[2] // 14
            )  # FIXME: 14 is hardcoded patch size

            images = self.transform.apply_image(images)  # preprocess images for sam
            resize = images.shape[:2]
            unique_label = np.unique(label).tolist()
            # 留不留
            if 255 in unique_label:
                unique_label.remove(255)
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
        for sampled_cls in sampled_classes:
            text = sampled_cls
            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list) # 随机
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))
            if ds in ["paco_lvis", "pascal_part"]:
                continue
            # 从文本映射为数字
            class_id = self.classes.tolist().index(sampled_cls) # 找到在整个class列表中的id
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
        
        # （H, W, C）-> (C, H, W)
        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
        label = torch.from_numpy(label).long()
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)
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

import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from PIL import Image

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
# from .reason_seg_dataset import ReasonSegDataset
# from .refer import REFER
# from .refer_seg_dataset import ReferSegDataset
from .seg_dataset import SemSegDataset
from .utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from .vqa_dataset import VQADataset


def collate_fn(batch, tokenizer=None):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    tokenize_data = tokenizer(
        conversation_list,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    input_ids = tokenize_data.input_ids
    attention_masks = tokenize_data.attention_mask

    IGNORE_TOKEN_ID = -100
    conv = get_default_conv_template("vicuna").copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            # if True:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            # rank0_print(tokenizer.decode(z))
            print(
                "conversation: ",
                conversation,
                "tokenizer.decode(z): ",
                tokenizer.decode(z),
            )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="VQA_RAD||PMC_VQA||SLAKE||Path_VQA",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference



# 读取class，image和mask路径
def init_slake_val(base_image_dir):
    folders = glob.glob(os.path.join(base_image_dir,"Slake/Slake1.0/imgs/*"))
    # print(folders)
    image_data = [os.path.join(f, "source.jpg") for f in folders]
    mask_data = [os.path.join(f, "mask.png") for f in folders]
    # class
    class_file = open(os.path.join(base_image_dir, "Slake/Slake1.0/mask.txt"))
    content = class_file.read()
    result = {}
    for line in content.split('\n'):
        key, value = line.split(':')
        result[int(key)] = value
    return (image_data, mask_data), result

# print(init_slake_val("/home/yangjinxia/mine/dataset/")[2])

class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        # splits = val_dataset.split("|")  # ReasonSeg|va
        # ds = val_dataset
        self.ds = val_dataset
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.data2list, self.data2class = eval("init_{}".format(self.ds))(base_image_dir)


    def __len__(self):
        
        return len(self.data2list[0])

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
        images, labels = self.data2list
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]
        label = Image.open(label_path)  # 1024*1024*3 此处有胸片  生成一下对应的mask图像？
        label = np.array(label)[:, :, 0]
        label[label == 0] = 255
        # json 文件是字典
        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")["pixel_values"][0]
        images = self.transform.apply_image(images)

        resize = images.shape[:2]
        unique_label = np.unique(label).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
        if len(unique_label) == 0 or len(unique_label) > 250:
            return self.__getitem__(0)  # 换张图
       
        sampled_classes_id = [i for i in unique_label if self.data2class.get(i) ]
        if len(sampled_classes_id) > 1:
            sampled_classes_id = np.random.choice(
                sampled_classes_id, size=1, replace=False
            ).tolist()
        classes = [self.data2class[i] for i in sampled_classes_id]
 
        conversations = []
        conv = get_default_conv_template("vicuna").copy()
        i = 0
        while i < len(classes): # 文本很长
            conv.messages = []
            text = classes[i].strip()
            conv.append_message(
                conv.roles[0],
                DEFAULT_IMAGE_TOKEN
                + " What is {} in this image? Please output segmentation mask.".format(
                    text
                ),
            )
            conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # replace <image> token
        image_token_len = 256
        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous()) # 前面只是把格式变了，还需后续处理
        
        label = torch.from_numpy(label)
        masks = []
        for class_id in sampled_classes_id:
            masks.append(label==class_id)
        masks = torch.stack(masks, dim=0) #1，1024，1024，3
        
        # masks = np.stack(masks, axis=0)
        # masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label  ###### 
        inference = True

        return (
            image_path,
            images,
            images_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )

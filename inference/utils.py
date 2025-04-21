import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch
import os
import json
from PIL import Image
import sys
import random

sys.path.append("llava/")
# sys.path.append("../")
from visual_prompt_organizer import vip_processor, visual_prompt_config


class QuestionDataset(Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]


class QuestionDataset_fromGTblank(Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]



class QuestionDataset_fromGTblank_vip_llava(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, args,processor,model_config):
        list_data_dict = json.load(open(args.question_file, "r"))

        self.list_data_dict = list_data_dict
        self.args = args
        self.processor=processor
        self.model_config=model_config

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]

            # if '-' in image_file:
            #     image_file=image_file.split('-')[0]
            # image_file = image_file.split('.')[0]+'/source.jpg'
            # conversation=sources[0]["conversations"]
            image = Image.open(
                os.path.join(self.args.image_folder, image_file)
            ).convert("RGB")
            if (
                type(sources[0]["id"]) == str
                and sources[0]["id"].split("-")[0] in visual_prompt_config
            ):
                try:
                    image, conversation = vip_processor(
                        sources[0],
                        image,
                        image_size_anchor=75,
                        data_args=self.args,
                    )
                    # print(conversation[0]['value'])
                except:
                    print("Fail in ViP image processing...")
                    return self.__getitem__(
                        random.randint(0, len(self.list_data_dict) - 1)
                    )
                sources[0]["conversations"] = conversation
            else:
                print(f"No images found!")
        question_info = {
            "id": sources[0]["id"],
            "gt_answer": sources[0]["conversations"][1]["value"],
            "metadata": sources[0]["metadata"],
            
        }
        if 'choice' in sources[0]:
            question_info['choice']=sources[0]['choice']
        # image_tensor = self.process_func([image], self.processor, self.model_config)[0]
        image_tensor=self.processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        return question_info, {"image": image_tensor, "conversation": conversation}


def setup():
    # os.environ["MASTER_ADDR"] = "localhost" # 由于这里是单机实验所以直接写 localhost
    # os.environ["MASTER_PORT"] = "12355"     # 任意空闲端口
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def cleanup():
    dist.destroy_process_group()


def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    return obj

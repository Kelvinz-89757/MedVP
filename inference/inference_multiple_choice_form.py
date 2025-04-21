import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append("../llava")
sys.path.append("../")
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import math
from transformers import set_seed, logging
import random
from utils import QuestionDataset_fromGTblank_vip_llava, setup, cleanup, tensor_to_serializable
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} started")

    set_seed(0)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=f"cuda:{rank}"
    )

    # questions = [
    #     json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    # ]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    if "llama-3" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    elif "phi-3" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end|>"),
        ]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]
    dataset = QuestionDataset_fromGTblank_vip_llava(
        args,
        processor=image_processor,
        model_config=model.config,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    # drop_last=False
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1,collate_fn=lambda x: x,num_workers=32)
    local_results = []

    # for ((question_info, image_info)) in tqdm(dataloader,position=0, file=sys.stdout):
    for batch in tqdm(dataloader,position=0, file=sys.stdout):
        # print(batch)
        question_info,image_info=batch[0]
        question_id = question_info["id"]
        idx = int(question_id.split("-")[2])
        image_tensor = image_info["image"]
        question = image_info["conversation"][0]["value"]
        gt_choice=question_info['choice']
        metadata=question_info['metadata']
        choice_ABCD2str='\n'.join([metadata['Choice A'],metadata['Choice B'],metadata['Choice C'],metadata['Choice D']])
        # print(image_tensor.shape)
        # print(question)
        gt_answer = question_info["gt_answer"]

        # qs = line["text"][0].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        qs = question.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        qs+=' Please choose from the following options:\n'+choice_ABCD2str+'\nYour reponse should be the choice letter(A, B, C, or D).'
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(rank)
        )

        # image = Image.open(os.path.join(args.image_folder, image_file))
        # image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            # "pixel_values"
        # ][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        if "Phi-3" in model_name:
            stop_str = "xsafs"
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(rank),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                eos_token_id=terminators,
                max_new_tokens=1024,
                use_cache=True,
            )

        input_token_len = input_ids.shape[1]
        # n_diff_input_output = (
        #     (input_ids != output_ids[:, :input_token_len]).sum().item()
        # )
        # if n_diff_input_output > 0:
        #     print(
        #         f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        #     )
        # outputs = tokenizer.batch_decode(
        #     output_ids[:, input_token_len:], skip_special_tokens=True
        # )[0]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        
        # check the format of the output
        if outputs not in ['A','B','C','D']:
            if len(outputs)==0:
                
                outputs=random.choice(['A','B','C','D'])
                tqdm.write(f'The output is not in the form of A, B, C, or D. The output is {outputs}, which is empty, so randomly selected from A, B, C, D')
            # 四个选项文本
            else:
                try:
                    choices = {
                        'A':metadata['Choice A'][metadata['Choice A'].index('A')+2:],
                        'B':metadata['Choice B'][metadata['Choice B'].index('B')+2:],
                        'C':metadata['Choice C'][metadata['Choice C'].index('C')+2:],
                        'D':metadata['Choice D'][metadata['Choice D'].index('D')+2:],
                    }
                except Exception as e:
                    print(f'Failure in extracting choices from metadata: {metadata}, with id {question_id}')
                    # raise RuntimeError(f"Error extracting choices from metadata with id {question_id}") from e
                    choices = {
                        'A':metadata['Choice A'],
                        'B':metadata['Choice B'],
                        'C':metadata['Choice C'],
                        'D':metadata['Choice D'],
                    }
                                    
                

                # 将 outputs 和选项文本放在一起进行向量化
                try:
                    texts = [outputs] + list(choices.values())
                    vectorizer = TfidfVectorizer().fit_transform(texts)
                    vectors = vectorizer.toarray()

                    # 计算输出文本与每个选项的余弦相似度
                    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

                    # 找到相似度最高的选项
                    best_match_index = np.argmax(cosine_similarities)
                    best_choice = list(choices.keys())[best_match_index]
                    
                    tqdm.write(f'The output is not in the form of A, B, C, or D. The output is {outputs}, use similarity to find the best match: {best_choice}, which is {choices[best_choice]}')
                    outputs=best_choice
                except:
                    outputs=random.choice(['A','B','C','D'])
                    tqdm.write(f'The output is not in the form of A, B, C, or D. The output is {outputs}, which is empty, so randomly selected from A, B, C, D')
        ans_id = shortuuid.uuid()
        
        metadata=question_info['metadata']
        # print(outputs)
        result = {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "gt_answer": gt_choice,
            "gt_answer_text": gt_answer,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": metadata,
        }
        serializable_result = tensor_to_serializable(result)

        local_results.append(serializable_result)

    dist.barrier()
    print(f"Rank {rank} reached barrier")
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)
    print(f"Rank {rank} finished all_gather_object")
    if rank == 0:
        all_results = [item for sublist in gathered_results for item in sublist]
        # all_results.sort(key=lambda x: x["question_id"])
        unique_results = []
        seen_ids = set()
        for result in all_results:
            if result["question_id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["question_id"])
        unique_results.sort(key=lambda x: x["question_id"])
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        with open(answers_file, "w") as ans_file:
            for res in unique_results:
                ans_file.write(json.dumps(res) + "\n")
        print(f"Rank {rank} finished writing to file {args.answers_file}")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--pwise", action='store_true', help="Description for pwise")
    parser.add_argument("--label_aug", action='store_true', help="Description for label_aug")
    parser.add_argument("--bboxPos_aug", action='store_true', help="Description for bPos_aug")
    parser.add_argument("--force_shape", type=str, default=None, help="Description for force_shape")
    args = parser.parse_args()

    eval_model(args)

import argparse
import json
import torch
import pprint
from tqdm import tqdm
import os
import torch.distributed as dist
import re
from itertools import chain
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import queue as queue_module
from queue import Empty


from utils import get_model_tokenizer, get_loaded_model_tokenizer, setup, cleanup, log_args, gather_and_merge_dicts, gather_and_merge_list
from utils_adastar import get_dataloader_adastar, update_minheap_winlose_front, log_stable_minheap_new

pp = pprint.PrettyPrinter(indent=2).pprint

def write_new_data(args, pred, data, endoftext):
    if args.task == "arc_challenge":
        q, choices = data["question"], data["choices"]
        options_text = "\n".join([f"{label}. {opt}" for label, opt in zip(choices["label"], choices["text"])])
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    elif args.task == "cqa":
        text = data["question"]
        q = text['stem']
        choices = text['choices']
        options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    elif args.task == "gsm8k":
        q = data["question"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "svamp":
        q = data["question_concat"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "cladder":
        q = data["question"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "anli_r1":
        t1 = data["premise"]
        choices = data['choices']
        labels = choices["label"]
        texts = choices["text"]
        options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])

        t2 = data["hypothesis"]
        q = f"{t1} {t2}"
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    else:
        raise NotImplementedError

    return new_example

def test_metric_STaR(args, predictions, datas, tokenizer, rank):
    correct_str = []
    correct_data_idx = []

    correct = [False for _ in range(len(predictions))]

    for idx, (pred, data) in enumerate(zip(predictions, datas), 1):
        cur_correct = False
        answer = data["answer"]

        q_start_idx = pred.find("Q: ")
        if q_start_idx != -1:
            pred = pred[:q_start_idx]

        if "####" in pred:
            parts = pred.split("####")
            if len(parts) > 1 and len(parts[1].split()) > 0:
                pred = parts[0] + "#### " + parts[1].split()[0]
            else:
                pred = parts[0] + "#### "

        if args.task == "arc_challenge":
            matches = list(re.finditer(r"\b(A|B|C|D)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None

        elif args.task == "cladder":
            matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
            pred_answer = matches[-1].lower() if matches else None

        elif args.task == "cqa":
            matches = list(re.finditer(r"\b(A|B|C|D|E)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None
        
        elif args.task == "svamp":
            matches = re.findall(r"-?\d+\.?\d*", pred)
            pred_answer = matches[-1] if matches else None
            ref_match = re.search(r"-?\d+\.?\d*", str(answer))
            ref_answer = ref_match.group(0) if ref_match else None
            if pred_answer == ref_answer:
                cur_correct = True

        elif args.task == "anli_r1":
            matches = list(re.finditer(r"\b(0|1|2)\b", pred))
            pred_answer = matches[-1].group(1) if matches else None

        else:
            matches = re.findall(r"-?\d+\.?\d*", pred)
            pred_answer = matches[-1] if matches else None
            ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer))
            ref_answer = ref_match.group(1).strip() if ref_match else None

            if pred_answer == ref_answer:
                cur_correct = True
                
        if args.task == "arc_challenge" and pred_answer and pred_answer == answer:
            cur_correct = True
        elif args.task == "cqa" and pred_answer and pred_answer == answer:
            cur_correct = True
        elif args.task == "cladder" and pred_answer and pred_answer == answer.lower():
            cur_correct = True
        elif args.task == "anli_r1" and pred_answer and str(pred_answer) == str(answer):
            cur_correct = True
        
        if cur_correct:
            correct[idx-1] = True
            correct_str.append(write_new_data(args, pred, data, tokenizer.eos_token))
            correct_data_idx.append(data["idx"])

    return correct_str, correct_data_idx, correct

    
def broadcast_list(data, src_rank):
    object_list = [data if dist.get_rank() == src_rank else None]
    dist.broadcast_object_list(object_list, src=src_rank)
    return object_list[0]

def prompt_preprocess(args, examples, tokenizer, n_shot_prompts, n_shot_prompts_hint, hint):
    combined_texts = []

    if args.task == "arc_challenge":
        for i in range(len(examples)):
            q, choices, a = examples[i]["question"], examples[i]["choices"], examples[i]["answerKey"]
            options_text = "\n".join([f"({label}). {opt}" for label, opt in zip(choices["label"], choices["text"])])
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nOptions:\n{options_text}\nA: ")

    elif args.task == "cqa":
        for i in range(len(examples)):
            q = examples[i]["question"]['stem']
            choices = examples[i]["question"]['choices']
            ans = examples[i]["answerKey"]
            options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])

            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({ans})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nOptions:\n{options_text}\nA: ")

    elif args.task == "gsm8k":
        for i in range(len(examples)):
            q = examples[i]["question"]
            a = examples[i]["answer"]
            answer = a.split()[-1]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({answer})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "svamp":
        for i in range(len(examples)):
            q = examples[i]["question_concat"]
            a = examples[i]["answer"]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "cladder":
        for i in range(len(examples)):
            q = examples[i]["question"]
            a = examples[i]["answer"]
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} ({a})\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q}\nA: ")

    elif args.task == "anli_r1":
        for i in range(len(examples)):
            q = examples[i]["premise"]
            h = examples[i]["hypothesis"]
            choices = examples[i]["choices"]
            a = examples[i]["label"]  
            labels = choices["label"]
            texts = choices["text"]
            options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])
            if hint[i] and args.no_hint == False:
                combined_texts.append(f"{n_shot_prompts_hint}\nQ: {q} {h} ({a})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{n_shot_prompts}\nQ: {q} {h}\nOptions:\n{options_text}\nA: ")

    else:
        raise NotImplementedError

    tokenized = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )
    
    return tokenized

def eval_examples(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint):
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate
    world_size = dist.get_world_size()

    correct_total = torch.zeros(4).to(rank)
    gpu_correct_total = torch.zeros(4).to(rank)

    pbar = tqdm(total=args.required_data_num, desc="Processing", disable=(rank != 0))

    gpu_rationale_dataset = []
    gpu_data_idx = []
    gpu_winlose = {}
    buffer = []
    hint = []
    data_depleted = torch.zeros(1).to(rank)

    def fill_buffer():
        while len(buffer) < args.batch_size:
            try:
                new_data = data_queue.get(timeout=2)
                if new_data is None:
                    break
                buffer.append(new_data)
                hint.append(False)
                id_idx = new_data["idx"]
                gpu_winlose[f'id_{id_idx}'] = {"iter": args.exp_iter, "win": 0, "total": 0}
            except (queue_module.Empty, EOFError, OSError) as e:
                break

    fill_buffer()

    batch_idx = 0
    
    with torch.no_grad():
        while True:
            fill_buffer()

            if len(buffer) == 0:
                data_depleted[0] = 1
            else:
                tokenized = prompt_preprocess(args, buffer, tokenizer, n_shot_prompts, n_shot_prompts_hint, hint)
                input_ids = tokenized["input_ids"].to(rank)
                attention_mask = tokenized["attention_mask"].to(rank)

                outputs = generate_fn(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p = 0.9,
                    temperature=args.inference_temp
                )

                generated_tokens = outputs[:, input_ids.shape[-1]:]
                predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                eos_token_id = tokenizer.eos_token_id
                actual_lengths = 0
                actual_c=0
                for seq in generated_tokens:
                    eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_positions.numel() > 0:
                        actual_length = eos_positions[0].item() + 1
                    else:
                        actual_length = seq.size(0)
                    actual_lengths+=actual_length
                    actual_c+=1

                actual_gen_len = actual_lengths
                
                flops_log_file = f"{args.idx_save}/flops_log_{rank}.json"
                log_args(flops_log_file, iter=args.exp_iter,idx=batch_idx, split="inf", hint=hint , batch=actual_c, input= actual_c*input_ids.size(1),output= actual_gen_len)
                batch_idx += 1

                correct_str, correct_data_idx, correct = test_metric_STaR(args, predictions, buffer, tokenizer, rank)
                
                for i in range(len(buffer)):
                    if hint[i]:
                        gpu_correct_total[3] += 1
                        if correct[i]:
                            gpu_correct_total[1] += 1
                    else:
                        gpu_correct_total[2] += 1
                        if correct[i]:
                            gpu_correct_total[0] += 1

                correct_total = gpu_correct_total.clone()
                gpu_rationale_dataset.extend(correct_str)
                gpu_data_idx.extend(correct_data_idx)

                new_buffer = []
                new_hint = []
                for buffer_idx in range(len(buffer)):
                    id_idx = buffer[buffer_idx]["idx"]
                    if correct[buffer_idx]:
                        gpu_winlose[f"id_{id_idx}"]["win"] += 1
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1
                    else:
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1                        
                        if not hint[buffer_idx]:
                            new_buffer.append(buffer[buffer_idx])
                            new_hint.append(True)
                buffer = new_buffer
                hint = new_hint

            dist.barrier()
            dist.all_reduce(correct_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(data_depleted, op=dist.ReduceOp.SUM)

            corrects = int(correct_total[0].item() + correct_total[1].item())
            if rank == 0:
                pbar.n = corrects
                pbar.refresh()

            if corrects >= args.required_data_num:
                break
            if data_depleted[0] == world_size:
                break

    if rank == 0:
        pbar.close()
    dist.barrier()
    return gpu_rationale_dataset, gpu_data_idx, gpu_winlose, correct_total

def eval_examples_nohint(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint):
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate
    world_size = dist.get_world_size()

    correct_total = torch.zeros(4).to(rank)
    gpu_correct_total = torch.zeros(4).to(rank)

    pbar = tqdm(total=args.required_data_num, desc="Processing", disable=(rank != 0))

    gpu_rationale_dataset = []
    gpu_data_idx = []
    gpu_winlose = {}
    buffer = []
    second = []
    data_depleted = torch.zeros(1).to(rank)

    def fill_buffer():
        while len(buffer) < args.batch_size:
            try:
                new_data = data_queue.get(timeout=2)
                if new_data is None:
                    break
                buffer.append(new_data)
                second.append(False)
                id_idx = new_data["idx"]
                gpu_winlose[f'id_{id_idx}'] = {"iter": args.exp_iter, "win": 0, "total": 0}
            except (queue_module.Empty, EOFError, OSError) as e:
                break

    fill_buffer()

    batch_idx = 0

    with torch.no_grad():
        while True:
            fill_buffer()

            if len(buffer) == 0:
                data_depleted[0] = 1
            else:
                queue_empty = fill_buffer()
                tokenized = prompt_preprocess(args, buffer, tokenizer, n_shot_prompts, n_shot_prompts_hint, second)
                input_ids = tokenized["input_ids"].to(rank)
                attention_mask = tokenized["attention_mask"].to(rank)

                outputs = generate_fn(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p = 0.9,
                    temperature=args.inference_temp
                )

                generated_tokens = outputs[:, input_ids.shape[-1]:]
                predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                eos_token_id = tokenizer.eos_token_id
                actual_lengths = 0
                actual_c=0
                for seq in generated_tokens:
                    eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_positions.numel() > 0:
                        actual_length = eos_positions[0].item() + 1
                    else:
                        actual_length = seq.size(0)
                    actual_lengths+=actual_length
                    actual_c+=1

                actual_gen_len = actual_lengths
                
                flops_log_file = f"{args.idx_save}/flops_log_{rank}.json"
                log_args(flops_log_file, iter=args.exp_iter,idx=batch_idx, split="inf", hint=second , batch=actual_c, input= actual_c*input_ids.size(1),output= actual_gen_len)
                batch_idx += 1

                correct_str, correct_data_idx, correct = test_metric_STaR(args, predictions, buffer, tokenizer, rank)
                
                for i in range(len(buffer)):
                    if second[i]:
                        if correct[i]:
                            gpu_correct_total[0] += 1
                    else:
                        gpu_correct_total[1] += 1
                        if correct[i]:
                            gpu_correct_total[0] += 1

                correct_total = gpu_correct_total.clone()
                gpu_rationale_dataset.extend(correct_str)
                gpu_data_idx.extend(correct_data_idx)

                new_buffer = []
                new_second = []
                for buffer_idx in range(len(buffer)):
                    id_idx = buffer[buffer_idx]["idx"]
                    if correct[buffer_idx]:
                        gpu_winlose[f"id_{id_idx}"]["win"] += 1
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1
                    else:
                        gpu_winlose[f"id_{id_idx}"]["total"] += 1                        
                        if not second[buffer_idx]:
                            new_buffer.append(buffer[buffer_idx])
                            new_second.append(True)
                buffer = new_buffer
                second = new_second

            dist.barrier()
            dist.all_reduce(correct_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(data_depleted, op=dist.ReduceOp.SUM)

            corrects = int(correct_total[0].item())
            if rank == 0:
                pbar.n = corrects
                pbar.refresh()

            if corrects >= args.required_data_num:
                break
            if data_depleted[0] == world_size:
                break

    if rank == 0:
        pbar.close()
    dist.barrier()
    return gpu_rationale_dataset, gpu_data_idx, gpu_winlose, correct_total

def evaluate(args, model, rank, world_size, data_queue, tokenizer, gen_length, target_save, n_shot_prompts, n_shot_prompts_hint):
    model.eval()

    if args.split == "train":
        if args.no_hint:
            gpu_rationale_dataset, gpu_data_idx, gpu_winlose, all_correct_total = eval_examples_nohint(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint)
        else:
            gpu_rationale_dataset, gpu_data_idx, gpu_winlose, all_correct_total = eval_examples(args, model, rank, data_queue, tokenizer, gen_length, n_shot_prompts, n_shot_prompts_hint)
    else:
        print("device_inference_adastar is not supposed to be called in dev split")
        raise NotImplementedError

    all_rationale_dataset = gather_and_merge_list(gpu_rationale_dataset, dst=0)
    all_data_idx = gather_and_merge_list(gpu_data_idx, dst=0)
    all_winlose = gather_and_merge_dicts(gpu_winlose, dst=0)

    return all_rationale_dataset, all_data_idx, all_winlose, all_correct_total

def get_ckpt_path(args, ckpt_step=-1):
    model_dir = args.model_dir
    if ckpt_step == -1:
        ckpt_step = args.total_steps    
        
    path = f"{model_dir}/step_{ckpt_step}/lm.pt"

    if args.split == "train" and ckpt_step == 0:
        print(f"Generate train set using base model")
        return "base_model"

    if os.path.exists(path):
        return path
    else: 
        raise FileNotFoundError(f"Model path {path} not found, exiting")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument('--rationalize', action='store_true', help="Whether to use rationalization")
    parser.add_argument('--show_hint_prompt', action='store_true', help="Whether a hint prompt will be necessary")
    parser.add_argument("--split", type=str, default="dev", help="Split")
    parser.add_argument("--task", type=str, default="cqa", help="Which dataset to run on")
    parser.add_argument("--ckpt_step", type=int, default=-1, help="Which checkpoint to eval. -1 means the final one")
    parser.add_argument("--exp_iter", type=int, default=-1, help="exp iteration for logging")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")
    parser.add_argument("--log_dir", type=str, default="", help="logging dir")
    parser.add_argument("--cur_total_steps", type=int, required=True, help="current total steps")
    parser.add_argument("--flops_dir", type=str, default="", help="logging dir")

    return parser.parse_args()

def fsdp_main(rank, world_size, args, data_queue):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    n_shot_prompts = ""
    n_shot_prompts_hint = ""
    prompt_file_path = f"./n_shot_prompts/{args.task}.json"
    prompt_hint_file_path = f"./n_shot_prompts/{args.task}_hint.json"
    with open(prompt_file_path, "r") as f:
        data = json.load(f)
    with open(prompt_hint_file_path, "r") as f:
        data_hint = json.load(f) 
    n_shot_prompts = [item["prompt"] for item in data["n_shot_prompts"]]
    n_shot_prompts_hint = [item["prompt"] for item in data_hint["n_shot_prompts"]]
    n_shot_prompts = "\n".join(n_shot_prompts)
    n_shot_prompts_hint = "\n".join(n_shot_prompts_hint)

    ckpt_path = get_ckpt_path(args, args.ckpt_step)
    if ckpt_path != "base_model": 
        dist.barrier()
        model, tokenizer = get_loaded_model_tokenizer(args, ckpt_path, args.model_name, rank, eval=True)
        dist.barrier()
        if rank == 0:
            print(f"[Inference {args.split}] model loaded from {ckpt_path}")
        
    else: 
        if args.base_model_path != None:
            dist.barrier()
            model, tokenizer = get_loaded_model_tokenizer(args, args.base_model_path, args.model_name, rank, eval=True)
            dist.barrier()
            if rank == 0:
                print(f"[Inference {args.split}] base model loaded from {args.base_model_path}")
        else:
            dist.barrier()
            model, tokenizer = get_model_tokenizer(args, args.model_name, rank, eval=True)
            dist.barrier()
            if rank == 0:
                print(f"[Inference {args.split}] base model path == None, using hf base model")

    tokenized_prompt = tokenizer(n_shot_prompts, return_tensors="pt")
    prompt_tokenized_len = tokenized_prompt["input_ids"].shape[1]
    tokenized_prompt_hint = tokenizer(n_shot_prompts_hint, return_tensors="pt")
    prompt_tokenized_len_hint = tokenized_prompt_hint["input_ids"].shape[1]
    args.max_length += max(prompt_tokenized_len, prompt_tokenized_len_hint)

    effective_batch_size = args.batch_size * args.grad_accumulation * world_size
    args.required_data_num = args.cur_total_steps * effective_batch_size
    if rank == 0:
        print(f"Required data num: {args.required_data_num}")
        
    args.batch_size = args.test_batch_size

    dataset_train, winlose, heap = get_dataloader_adastar(args, tokenizer, rank=rank, world_size=world_size)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    if args.split == "train":
        init_start_event.record()
        all_rationale_dataset, all_data_idx, all_winlose, all_correct_total = evaluate(args, model, rank, world_size, data_queue, tokenizer, args.gen_length, args.target_save, n_shot_prompts, n_shot_prompts_hint)
        init_end_event.record()
        if rank == 0:
            log_args(f"{args.log_dir}/elapsed_time_log.json", iter=args.exp_iter, log_point="gen_train_data", time=init_start_event.elapsed_time(init_end_event) / 1000)
    else:
        print("device_inference_adastar is not supposed to be called in dev split")
        raise NotImplementedError

    dist.barrier()
    if rank == 0:
        if args.no_hint:
            nohint_correct = int(all_correct_total[0].item())
            nohint_total = int(all_correct_total[1].item())
            
            accuracy = nohint_correct / nohint_total
            hint_accuracy = "_"

            total_inference = nohint_total
            total_accuracy = accuracy
        else:
            nohint_correct = int(all_correct_total[0].item())
            hint_correct = int(all_correct_total[1].item())
            nohint_total = int(all_correct_total[2].item())
            hint_total = int(all_correct_total[3].item())

            accuracy = nohint_correct / nohint_total
            hint_accuracy = hint_correct / hint_total

            total_inference = nohint_total
            total_accuracy = accuracy + (1 - accuracy) * hint_accuracy

        num_used_data = args.required_data_num
        num_pop_data = round(total_inference * (total_accuracy ** args.accuracy_power))
        
        updated_minheap, updated_winlose = update_minheap_winlose_front(heap, winlose, all_winlose, num_pop_data=num_pop_data)

        for new_example, data_idx in zip(all_rationale_dataset[:num_used_data], all_data_idx[:num_used_data]):
            with open(args.target_save + "/correct_data.txt", 'a+') as new_train_f:
                print(new_example, file=new_train_f, end="\n\n")

            new_example_no_answer = "A:".join(new_example.split("A:")[:-1])
            with open(args.idx_save + f"/{args.split}_corr_idx_{args.exp_iter}.txt", 'a+') as new_idx_f:
                print(f"idx: {data_idx}\n{new_example_no_answer}", file=new_idx_f, end="\n\n")

        json.dump(updated_winlose, open(args.log_dir + "/winlose.json", "w"))
        updated_minheap.save(args.log_dir + "/heap.pkl")


        log_args(f"{args.log_dir}/eval_log.json", iter=args.exp_iter, split=args.split, accuracy=accuracy, hint_accuracy=hint_accuracy)
        json.dump(updated_winlose, open(args.target_save + "/winlose.json", "w"))
        log_stable_minheap_new(updated_minheap, args.target_save + "/heap_stats.json")
    cleanup()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    print(args)
    split = args.split
    params = json.load(open(args.config))

    args.batch_size = params["batch_size"]
    args.test_batch_size = params["test_batch_size"]
    args.model_name = params["model_name"]
    args.precision = params["precision"]
    args.max_length = params["max_length"]
    args.gen_length = params["gen_length"]
    args.n_shot = params["n_shot"]
    args.self_consistency = params["self_consistency"]
    args.delete_model_after_loading = params["delete_model_after_loading"]
    args.lora = params.get("lora", None)
    args.grad_accumulation = params["grad_accumulation"]
    args.inference_temp = params["inference_temp"]
    args.no_hint = params["no_hint"]
    args.base_model_path = params["base_model_path"]

    args.name = params["name"]
    args.idx_save = params["target_save"] 
    args.target_save = params["target_save"] if split != "dev" else f'{args.task}/new_dev.txt'
    args.model_dir = params["model_dir"]
    args.total_steps = params.get("total_steps", 0)
    
    args.method = params["method"]
    args.accuracy_power = params["accuracy_power"]

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset_train, winlose, heap = get_dataloader_adastar(args, tokenizer, rank=0, world_size=WORLD_SIZE)

    with mp.Manager() as manager:
        queue = manager.Queue()
        for item in dataset_train:
            queue.put(item)
        for _ in range(args.test_batch_size * WORLD_SIZE):
            queue.put(None)
        mp.spawn(fsdp_main, args=(WORLD_SIZE, args, queue), nprocs=WORLD_SIZE, join=True)



import os
import json
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType, FullStateDictConfig, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import pickle
import logging
from safetensors.torch import load_file
import glob


def build_problem_id_mapping(args, dataset_name, tokenizer=None):
    problem_mapping = {}

    try:
        if dataset_name == "gsm8k":
            dataset = load_from_disk("../datasets/data_gsm8k")
            train_data = dataset["train"]
            test_data = dataset["test"]

            for idx, example in enumerate(train_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "train"
                }

            for idx, example in enumerate(test_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "test"
                }

        elif dataset_name == "cladder":
            train_data = load_dataset("json", data_files="./datasets/data_cladder_split/cladder_train_long.jsonl")[
                "train"]
            test_data = load_dataset("json", data_files="./datasets/data_cladder_split/cladder_test.jsonl")["train"]

            for idx, example in enumerate(train_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "train"
                }

            for idx, example in enumerate(test_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "test"
                }

        elif dataset_name == "arc_challenge":
            train_data = load_dataset("json", data_files="./datasets/data_arc_challenge/train_val_combined.jsonl")[
                "train"]
            test_data = load_dataset("json", data_files="./datasets/data_arc_challenge/test_new.jsonl")["train"]

            for idx, example in enumerate(train_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "train"
                }

            for idx, example in enumerate(test_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "test"
                }

        elif dataset_name == "cqa":
            train_data = load_dataset("json", data_files="./datasets/CommonsenseQA/train_rand_split.jsonl")["train"]
            test_data = load_dataset("json", data_files="./datasets/CommonsenseQA/test.jsonl")["train"]

            for idx, example in enumerate(train_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "train"
                }

            for idx, example in enumerate(test_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "test"
                }

        elif dataset_name == "anli_r1":
            train_data = load_dataset("json", data_files="./datasets/data_anli/train_r1_modified.jsonl")["train"]
            test_data = load_dataset("json", data_files="./datasets/data_anli/test_r1_modified.jsonl")["train"]

            for idx, example in enumerate(train_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "train"
                }

            for idx, example in enumerate(test_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "test"
                }

        elif dataset_name == "svamp":
            train_data = load_dataset("json", data_files="./datasets/data_svamp/train.jsonl")["train"]
            test_data = load_dataset("json", data_files="./datasets/data_svamp/test.jsonl")["train"]

            for idx, example in enumerate(train_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "train"
                }

            for idx, example in enumerate(test_data):
                question_id = str(example.get("idx", idx))
                problem_mapping[question_id] = {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "split": "test"
                }

        elif dataset_name == "STaR":
            if not hasattr(args, 'dataset_path'):
                raise ValueError("dataset_path is required for STaR dataset")

            with open(args.dataset_path, 'rb') as f:
                dataset_train = pickle.load(f)

            for item in dataset_train:
                idx = str(item.get('idx'))
                if 'tokens' in item and tokenizer:
                    text = tokenizer.decode(item['tokens'], skip_special_tokens=True)
                    parts = text.split('Answer:')
                    question = parts[0].strip() if len(parts) > 0 else text
                    answer = parts[1].strip() if len(parts) > 1 else ""

                    problem_mapping[idx] = {
                        "question": question,
                        "answer": answer,
                        "original_text": text,
                        "split": "train"  
                    }
                else:
                    problem_mapping[idx] = item

        else:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Add specific handling for this dataset.")

        print(f"Successfully built mapping for {len(problem_mapping)} problems from {dataset_name}")
        return problem_mapping

    except Exception as e:
        print(f"Error building problem mapping for {dataset_name}: {e}")
        return {}

def initialize_problem_tracking(args, dataset):
    tracking_file = os.path.join(args.log_dir, "problem_tracking.json")

    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        if args.rank == 0:
            print(f"Problem tracking data loaded from {tracking_file}")
    else:
        tracking_data = {}

        if isinstance(dataset, dict) and "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset

        for idx, example in enumerate(train_data):
            question_id = example.get("idx", idx)  
            tracking_data[str(question_id)] = 0

        if args.rank == 0:
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f, indent=2)
            print(f"Problem tracking data initialized and saved to {tracking_file}")

    return tracking_data


def get_problem_tracking_stats(args, cur_iter=None, dataset=None):
    tracking_file = os.path.join(args.log_dir, "problem_tracking.json")

    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)

    iter_counts = {}
    for problem_id, last_iter in tracking_data.items():
        iter_counts[last_iter] = iter_counts.get(last_iter, 0) + 1

    never_solved = iter_counts.get(0, 0)

    stats = {
        "total_problems": len(tracking_data),
        "solved_at_least_once": len(tracking_data) - never_solved,
        "never_solved": never_solved,
        "iteration_distribution": iter_counts
    }

    if cur_iter is not None and hasattr(args, 'sup_thresholds'):
        not_consistently_solved_ids = []

        for problem_id, last_iter in tracking_data.items():
            iterations_since_correct = cur_iter - last_iter if last_iter > 0 else cur_iter

            if iterations_since_correct >= args.sup_thresholds:
                not_consistently_solved_ids.append(problem_id)

        stats["not_consistently_solved_count"] = len(not_consistently_solved_ids)
        stats["not_consistently_solved_ids"] = not_consistently_solved_ids

        log_not_consistently_solved_problems(args, not_consistently_solved_ids, cur_iter, dataset)

    return stats


def log_not_consistently_solved_problems(args, problem_ids, cur_iter, dataset=None):
    if not problem_ids:
        print(f"All problems are consistently solved (sup_threshold={args.sup_thresholds}) at iteration {cur_iter}")
        return

    log_dir = args.log_dir if hasattr(args, 'log_dir') else f"{args.task}/{args.experiment_name}"
    log_file = os.path.join(log_dir, f"not_consistently_solved_problems_iter{cur_iter}.json")

    problem_details = []
    if dataset is not None:
        if isinstance(dataset, dict) and "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset

        id_to_index = {}
        for idx, example in enumerate(train_data):
            question_id = str(example.get("idx", idx))
            id_to_index[question_id] = idx

        for problem_id in problem_ids:
            if problem_id in id_to_index:
                idx = id_to_index[problem_id]
                example = train_data[idx]
                problem_details.append({
                    "id": problem_id,
                    "question": example.get("question", ""),
                    "answer": example.get("answer", "")
                })
            else:
                problem_details.append({"id": problem_id})
    else:
        problem_details = [{"id": pid} for pid in problem_ids]

    log_data = {
        "iteration": cur_iter,
        "sup_threshold": args.sup_thresholds,
        "not_consistently_solved_count": len(problem_ids),
        "not_consistently_solved_problems": problem_details
    }

    if args.rank == 0: 
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Logged {len(problem_ids)} not consistently solved problems to {log_file}")

def update_problem_tracking_stats(args, cur_iter, dataset=None):
    try:
        updated_stats = get_problem_tracking_stats(args, cur_iter, dataset)

        stats_file = os.path.join(f"{args.task}/{args.experiment_name}", f"problem_tracking_stats_iter{cur_iter}.json")
        if args.rank == 0:
            with open(stats_file, 'w') as f:
                json.dump(updated_stats, f, indent=2)
            print(f"Problem tracking stats updated for iteration {cur_iter}")

            if "not_consistently_solved_count" in updated_stats:
                print(f"Found {updated_stats['not_consistently_solved_count']} problems NOT solved consistently "
                      f"(sup_threshold={args.sup_thresholds})")
    except Exception as e:
        print(f"Error updating problem tracking stats: {e}")

def update_problem_tracking(args, correct_problems, current_iter):
    tracking_file = os.path.join(args.log_dir, "problem_tracking.json")
    with open(tracking_file, 'r') as f:
        tracking_data = json.load(f)
    for problem_id in correct_problems:
        tracking_data[str(problem_id)] = current_iter
    if args.rank == 0:
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        print(f"Problem tracking data updated for {len(correct_problems)} problems at iteration {current_iter}")
    return tracking_data

def merge_flops_logs(args):
    if args.split=='dev':
        flops_log_files = glob.glob(f"{args.flops_dir}/flops_log_*.json")
    elif args.split=='train':
        flops_log_files = glob.glob(f"{args.idx_save}/flops_log_*.json")
    else:
        flops_log_files = glob.glob(f"{args.log_dir}/flops_log_*.json")
    merged_data = []

    for file in flops_log_files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                merged_data.extend(data)
            except json.JSONDecodeError:
                continue
    if args.split=='dev':
        with open(f"{args.flops_dir}/flops_log.json", 'w') as f:
            json.dump(merged_data, f, indent=2)
    elif args.split=='train':
        with open(f"{args.idx_save}/flops_log.json", 'w') as f:
            json.dump(merged_data, f, indent=2)
    else:
        with open(f"{args.log_dir}/flops_log.json", 'w') as f:
            json.dump(merged_data, f, indent=2)


def fsdp_wrap(args, model, rank, cpu_offload=False):
    if args.precision == "bf16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,  
            reduce_dtype=torch.bfloat16,  
            buffer_dtype=torch.bfloat16  
            )
    elif args.precision == "fp16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16, 
            reduce_dtype=torch.float16, 
            buffer_dtype=torch.float16 
            )

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    args.cfg = cfg

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload,
        device_id=torch.device(rank), 
        )
    return model


def get_loaded_model_tokenizer(args, model_path, model_name, rank, cpu_offload=False, eval=False):
    if args.precision == "bf16":
        model_dtype = torch.bfloat16

    if model_name == "Qwen/Qwen2.5-3B" :
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,    attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    else :
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
    if not args.lora:
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        if eval:
            model = model.to(rank)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            model = fsdp_wrap(args, model, rank, cpu_offload=cpu_offload)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora["lora_rank"],
            lora_alpha=args.lora["lora_alpha"],
            lora_dropout=args.lora["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model = model.to(model_dtype)
        if eval:
            model = model.to(rank)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            model = fsdp_wrap(args, model, rank)

        lora_state_dict = torch.load(model_path, map_location="cpu")
        model_state_dict = model.state_dict()
        updated_lora_state_dict = {}
        for key in lora_state_dict.keys():
            new_key = "module." + key if "module." + key in model_state_dict else key
            updated_lora_state_dict[new_key] = lora_state_dict[key]

        model.load_state_dict(updated_lora_state_dict, strict=False) 
        dist.barrier()

        if rank == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.print_trainable_parameters()
            else:
                model.print_trainable_parameters()

    model = model.to(rank)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def get_model_tokenizer(args, model_name, rank, eval=False):
    if args.precision == "bf16":
        model_dtype = torch.bfloat16
    if model_name == "Qwen/Qwen2.5-3B" :
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,    attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    else :
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora["lora_rank"],
            lora_alpha=args.lora["lora_alpha"],
            lora_dropout=args.lora["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)
        model = model.to(model_dtype)

        if rank == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.print_trainable_parameters()
            else:
                model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if eval:
        model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model = fsdp_wrap(args, model, rank)
    return model, tokenizer


def get_optimizer_scheduler_step_based(args, model, train_loader):
    if args.task == "gsm8k":
        if args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        if args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

        def lr_lambda(current_step: int):
            if current_step < args.warm_up_steps:
                return float(current_step) / float(max(1, args.warm_up_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

def get_dataloader_STaR(args, tokenizer, rank, world_size):
    with open(args.dataset_path, 'rb') as f:
        dataset_train = pickle.load(f)

    if rank == 0:
        print(f"Dataset from {args.dataset_path} : total {len(dataset_train)} sequences")

    processed_data = []
    for item in dataset_train:
        processed_data.append({
            "tokens": item['tokens'],
            "idx": item['idx'],
            "original_position": len(processed_data) 
        })

    dataset_train = Dataset.from_list(processed_data)

    dataset_train = dataset_train.map(
        lambda examples: preprocess_function_STaR(args, examples, tokenizer),
        batched=True
    )

    sampler_train = DistributedSampler(
        dataset_train,
        rank=rank,
        num_replicas=world_size,
        shuffle=False
    )

    train_kwargs = {
        'batch_size': args.batch_size,
        'sampler': sampler_train,
        'collate_fn': collate_fn_with_metadata
    }

    cuda_kwargs = {
        'num_workers': 4,
        'pin_memory': True,
        'shuffle': False
    }

    train_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)

    return train_loader, sampler_train


def preprocess_function_STaR(args, examples, tokenizer):
    tokens_list = examples["tokens"]
    indices = examples["idx"]
    original_positions = examples["original_position"]

    tokenized = {
        "input_ids": [],
        "attention_mask": [],
        "idx": [],
        "original_position": []
    }

    for tokens, idx, orig_pos in zip(tokens_list, indices, original_positions):
        padded_tokens = tokenizer.pad(
            {"input_ids": [tokens]},
            max_length=args.max_length,
            padding="max_length",
            return_attention_mask=True
        )

        tokenized["input_ids"].append(padded_tokens["input_ids"][0])
        tokenized["attention_mask"].append(padded_tokens["attention_mask"][0])
        tokenized["idx"].append(idx)
        tokenized["original_position"].append(orig_pos)

    return tokenized


def collate_fn_with_metadata(batch):
    collated = {
        'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        'idx': [item['idx'] for item in batch],
        'original_position': [item['original_position'] for item in batch]
    }
    return collated

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSONL file: {e}")
    return data

def remove_matching_questions(dataset, questions_to_remove):
    filtered_dataset = dataset.filter(lambda example: example['question'] not in questions_to_remove)
    file_path=f"../removed/data_gsm8k_escape_filtered_{args.exp_iter-2}.json"
    filtered_dataset.to_json(file_path)

    return filtered_dataset

def get_wrong_examples_dataloader_STaR(args, wrong_examples, rank, world_size):
    dataset_test = Dataset.from_list(wrong_examples)

    sampler_test = DistributedSampler(dataset_test, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True)

    train_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler_test, 'collate_fn': collate_fn}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': False}

    train_kwargs.update(cuda_kwargs)

    test_loader = torch.utils.data.DataLoader(dataset_test, **train_kwargs)
    return test_loader, sampler_test


def preprocess_function_STaR(args, examples, tokenizer):
    examples = examples["tokens"]  

    tokenized = {"input_ids": [], "attention_mask": [], "text": []}

    for tokens in examples:
        original_text = tokenizer.decode(tokens, skip_special_tokens=True)

        padded_tokens = tokenizer.pad(
            {"input_ids": [tokens]},  
            max_length=args.max_length,
            padding="max_length", 
            return_attention_mask=True 
        )

        tokenized["input_ids"].append(padded_tokens["input_ids"][0])
        tokenized["attention_mask"].append(padded_tokens["attention_mask"][0])
        tokenized["text"].append(original_text)

    return tokenized

def add_idx_to_batch(batch, indices):
    batch["idx"] = indices
    return batch

def get_dataloader(args, tokenizer, rank, world_size):
    dataset, dataset_train, dataset_test = None, None, None
    if args.task == "gsm8k":
        dataset = load_from_disk("../datasets/data_gsm8k")
        dataset_test = load_dataset("json", data_files="./datasets/data_gsm8k/test.jsonl")["train"]

        dataset_train = dataset["train"].map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif (args.task == "cladder"):
        dataset_train = load_dataset("json", data_files="./datasets/data_cladder_split/cladder_train_long.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_cladder_split/cladder_test.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "arc_challenge":
        dataset_train = load_dataset("json", data_files="./datasets/data_arc_challenge/train_val_combined.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_arc_challenge/test_new.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "cqa":
        dataset_train = load_dataset("json", data_files="./datasets/CommonsenseQA/train_rand_split.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/CommonsenseQA/test.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "anli_r1":
        dataset_train = load_dataset("json", data_files="./datasets/data_anli/train_r1_modified.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_anli/test_r1_modified.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "svamp":
        dataset_train = load_dataset("json", data_files="./datasets/data_svamp/train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_svamp/test.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    dataset_train = dataset_train.map(
    add_idx_to_batch,
    with_indices=True,
    batched=True
    )

    dataset_test = dataset_test.map(
        add_idx_to_batch,
        with_indices=True,
        batched=True
    )

    sampler_train = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_test = DistributedSampler(dataset_test, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler_train, 'collate_fn': collate_fn}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler_test, 'collate_fn': collate_fn}
    cuda_kwargs = {'num_workers': 4,
                   'pin_memory': True,
                   'shuffle': False}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    return train_loader, sampler_train, test_loader, sampler_test

def preprocess_function(args, examples, tokenizer, split):
    if args.task == "arc_challenge":
        combined_texts = []

        for q, choices, ans in zip(examples["question"], examples["choices"], examples.get("answerKey", [""] * len(examples["question"]))):
            options_text = "\n".join([f"{label}. {opt}" for label, opt in zip(choices["label"], choices["text"])])
            if split == "train":
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        if split == "train":
            if "answerKey" not in examples:
                raise KeyError(f"Error: 'answerKey' field is missing in dataset! Available keys: {examples.keys()}")
            tokenized["labels"] = [ord(ans) - ord("A") for ans in examples["answerKey"]]

        tokenized["question"] = examples["question"]
        tokenized["options"] = [c["text"] for c in examples["choices"]]
        tokenized["answer"] = examples["answerKey"]

    elif args.task == "cqa":
        combined_texts = []

        for text, ans in zip(examples["question"], examples["answerKey"]):
            q = text['stem']
            choices = text['choices']
            options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
            if split == "train":
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length
        )

        tokenized["question"] = examples["question"]
        tokenized["answer"] = examples["answerKey"]

    elif args.task == "svamp":
        if split == "train":
            combined_texts = [f"Q: {q}\nA: {eq}\n#### {a}" for q, eq, a in zip(examples["question_concat"], examples["Equation"], examples["answer"])]
        else:
            combined_texts = [f"Q: {q}\nA: " for q in examples["question_concat"]]

        tokenized = tokenizer(
            combined_texts,
            padding="max_length", 
            truncation=True,
            max_length=args.max_length, 
        )

        tokenized["question"] = examples["question_concat"]
        tokenized["rationale"] = examples["Equation"]
        tokenized["answer"] = examples["answer"]

    elif args.task == "anli_r1":
        combined_texts = []

        for q,h, choices, ans in zip(examples["premise"], examples["hypothesis"],examples["choices"] ,examples["label"]):
            labels = choices["label"] 
            texts = choices["text"]  
            options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])
            if split == "train":
                combined_texts.append(f"Q: {q} {h}\nOptions:\n{options_text}\nA: {ans}")
            else:
                combined_texts.append(f"Q: {q} {h}\nOptions:\n{options_text}\nA: ")

        tokenized = tokenizer(
            combined_texts,
            padding="max_length", 
            truncation=True, 
            max_length=args.max_length, 
        )
        tokenized["question"] = [p + " " + h for p, h in zip(examples["premise"], examples["hypothesis"])]
        tokenized["rationale"] = examples["reason"]
        tokenized["answer"] = examples["label"]

    else:
        if split == "train":
            combined_texts = [f"Q: {q}\nA: {a}" for q, a in zip(examples["question"], examples["answer"])]
        else:
            combined_texts = [f"Q: {q}\nA: " for q in examples["question"]]

        tokenized = tokenizer(
            combined_texts,
            padding="max_length",  
            truncation=True, 
            max_length=args.max_length, 
        )

        tokenized["question"] = examples["question"]
        tokenized["answer"] = examples["answer"]

    return tokenized

def collate_fn(batch):
    collated_batch = {}

    for key in batch[0]:
        if key in ["input_ids", "attention_mask", "wrong_input_ids", "wrong_attention_mask"]:
            collated_batch[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        else:
            collated_batch[key] = [item[key] for item in batch]

    return collated_batch



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def log_args(output_file, **kwargs):
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as json_file:
                logs = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
    else:
        logs = []

    logs.append(kwargs)
    with open(output_file, "w") as json_file:
        json.dump(logs, json_file, indent=4)

def gather_and_merge_list(data_list, dst=0):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gathered_data = [None] * world_size  

    dist.all_gather_object(gathered_data, data_list)

    if rank == dst:
        all_dicts = [d for sublist in gathered_data for d in sublist]

        return all_dicts
    else:
        return None 

def gather_and_merge_dicts(local_dict, dst=0):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    gather_list = [None] * world_size if rank == dst else []

    dist.gather_object(obj=local_dict, object_gather_list=gather_list, dst=dst)

    if rank == dst:
        merged_dict = {}
        for d in gather_list:
            for k, v in d.items():
                if k not in merged_dict:
                    merged_dict[k] = v
        return merged_dict

    else:
        return None

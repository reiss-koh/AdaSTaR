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

from utils import get_model_tokenizer, get_loaded_model_tokenizer, get_wrong_examples_dataloader_STaR, get_dataloader, setup, cleanup, log_args, merge_flops_logs

pp = pprint.PrettyPrinter(indent=2).pprint


def write_new_data(args, target_save, pred, data, endoftext):
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
    elif args.task == "anli_r1":
        t1 = data["premise"]
        choices = data['choices']
        labels = choices["label"]
        texts = choices["text"]
        options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])

        t2 = data["hypothesis"]
        q = f"{t1} {t2}"
        new_example = f"Q: {q}\nOptions:\n{options_text}\nA: {pred}" + endoftext
    elif args.task == "svamp":
        q = data["question_concat"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    elif args.task == "cladder":
        q = data["question"]
        new_example = f"Q: {q}\nA: {pred}" + endoftext
    else:
        raise NotImplementedError

    new_example_no_answer = q
    with open(args.idx_save + f"/{args.split}_corr_idx_{args.exp_iter}.txt", 'a+') as new_idx_f:
        print(f"idx: {data['idx']}\nQ: {new_example_no_answer}", file=new_idx_f, end="\n\n")

    with open(target_save, 'a+') as new_train_f:
        print(new_example, file=new_train_f, end="\n\n")

    return new_example
    


def test_metric_STaR(args, predictions, datas, target_save, tokenizer, hint):
    wrong_examples = []
    correct, total = 0, 0

    try:
        for idx, (pred, data) in enumerate(zip(predictions, datas), 1):
            try:
                cur_correct = False
                answer = data.get("answer")
                if answer is None:
                    print(f"Warning: Missing answer for index {idx}")
                    continue

                q_start_idx = pred.find("Q: ")
                if q_start_idx != -1:
                    pred = pred[:q_start_idx]

                if "####" in pred:
                    parts = pred.split("####")
                    if len(parts) > 1 and len(parts[1].split()) > 0:
                        pred = parts[0] + "#### " + parts[1].split()[0]
                    else:
                        pred = parts[0] + "#### "

                pred_answer = None
                try:
                    if args.task == "arc_challenge":
                        matches = list(re.finditer(r"\b(A|B|C|D)\b", pred))
                        pred_answer = matches[-1].group(1) if matches else None
                    
                    elif args.task == "anli_r1":
                        matches = list(re.finditer(r"\b(0|1|2)\b", pred))
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
                    
                    else: 
                        matches = re.findall(r"-?\d+\.?\d*", pred)
                        pred_answer = matches[-1] if matches else None
                        ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer))
                        ref_answer = ref_match.group(1).strip() if ref_match else None

                        if pred_answer == ref_answer:
                            cur_correct = True

                except IndexError as e:
                    print(f"Warning: Failed to extract answer from prediction at index {idx}: {e}")
                    continue

                if args.task == "arc_challenge" and pred_answer and pred_answer == answer:
                    cur_correct = True
                elif args.task == "cqa" and pred_answer and pred_answer == answer:
                    cur_correct = True
                elif args.task == "anli_r1" and pred_answer and str(pred_answer) == str(answer):
                    cur_correct = True
                elif args.task == "cladder" and pred_answer and pred_answer == answer.lower():
                    cur_correct = True

                if cur_correct:
                    correct += 1
                    try:
                        if args.split == "train":
                            write_new_data(args, target_save + "/correct_data.txt", pred, data, tokenizer.eos_token)
                        else:
                            write_new_data(args, target_save, pred, data, tokenizer.eos_token)
                    except Exception as e:
                        print(f"Warning: Failed to write new data at index {idx}: {e}")
                else:
                    if not hint:
                        wrong_examples.append(data)
                total += 1

            except Exception as e:
                print(f"Warning: Error processing prediction at index {idx}: {e}")
                continue

    except Exception as e:
        print(f"Critical error in test_metric_STaR: {e}")

    return wrong_examples, correct, total

def eval_examples(args, model, rank, test_loader, tokenizer, gen_length, n_shot_prompts, hint=False):
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate

    eval_progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"{'Hint' if hint else 'No Hint'} Eval [Rank {rank}]",
        position=rank + 1,
        leave=False,
        disable=(rank != 0),
    )

    correctsum = 0
    totalsum = 0
    wrong_datasets = []

    with torch.no_grad():
        for batch_idx, data in eval_progress_bar:
            try:
                tokenized = prompt_preprocess(args, data, tokenizer, n_shot_prompts, hint=hint)
                input_ids = tokenized["input_ids"].to(rank)
                attention_mask = tokenized["attention_mask"].to(rank)

                try:
                    outputs = generate_fn(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_length,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_p=0.9,
                        temperature=args.inference_temp
                    )
                         
                except Exception as e:
                    print(f"Warning: Generation failed for batch {batch_idx}: {e}")
                    continue

                try:
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
          
                except Exception as e:
                    print(f"Warning: Decoding failed for batch {batch_idx}: {e}")
                    continue

                all_predictions = [None for _ in range(dist.get_world_size())]
                all_data = [None for _ in range(dist.get_world_size())]

                try:
                    dist.all_gather_object(all_predictions, predictions)
                    dist.all_gather_object(all_data, data)
                except Exception as e:
                    print(f"Warning: All-gather failed for batch {batch_idx}: {e}")
                    continue

                if rank == 0:
                    try:
                        all_predictions = list(chain.from_iterable(all_predictions))
                        merged_data = []
                        for rank_data in all_data:
                            for i in range(len(rank_data["question"])):
                                single_data = {}
                                for key in rank_data.keys():
                                    single_data[key] = rank_data[key][i]
                                merged_data.append(single_data)

                        wrong_examples, correct, total = test_metric_STaR(
                            args, all_predictions, merged_data,
                            args.target_save, tokenizer, hint=hint
                        )
                        correctsum += correct
                        totalsum += total
                        if not hint:
                            wrong_datasets.extend(wrong_examples)
                    except Exception as e:
                        print(f"Warning: Processing results failed for batch {batch_idx}: {e}")
                        continue

                dist.barrier()

            except Exception as e:
                print(f"Warning: Failed to process batch {batch_idx}: {e}")
                continue
    
    if rank == 0:
        if totalsum > 0:
            if hint:
                print(f"Hint Correct: {correctsum}, Accuracy: {correctsum / totalsum:.4f}")
            else:
                print(f"No hint Correct: {correctsum}, Accuracy: {correctsum / totalsum:.4f}")
        else:
            print("Warning: No valid examples were processed")
        
    return wrong_datasets, correctsum, totalsum 

def broadcast_list(data, src_rank):
    object_list = [data if dist.get_rank() == src_rank else None]
    dist.broadcast_object_list(object_list, src=src_rank)
    return object_list[0]

def prompt_preprocess(args, examples, tokenizer, prompt, hint):
    if args.task == "arc_challenge": 
        combined_texts = []
        for q, choices, a in zip(examples["question"], examples["choices"], examples["answerKey"]):
            options_text = "\n".join([f"({label}). {opt}" for label, opt in zip(choices["label"], choices["text"])])
            if hint:
                combined_texts.append(f"{prompt}\nQ: {q} ({a})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{prompt}\nQ: {q}\nOptions:\n{options_text}\nA: ")

    elif args.task == "cqa":
        combined_texts = []
        for text, ans in zip(examples["question"], examples["answerKey"]):
            q = text['stem']
            choices = text['choices']
            options_text = "\n".join([f'({choice["label"]}) {choice["text"]}' for choice in choices])
            if hint:
                combined_texts.append(f"{prompt}\nQ: {q} ({ans})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{prompt}\nQ: {q}\nOptions:\n{options_text}\nA: ")

    elif args.task == "anli_r1":
        combined_texts = []
        for q, h, a, choices in zip(examples["premise"], examples["hypothesis"], examples["label"],examples["choices"]):
            labels = choices["label"]  # [0, 1, 2]
            texts = choices["text"]    # ["entailment", "neutral", "contradiction"]
            options_text = "\n".join([f'({label}) {text}' for label, text in zip(labels, texts)])
            if hint:
                combined_texts.append(f"{prompt}\nQ: {q} {h} ({a})\nOptions:\n{options_text}\nA: ")
            else:
                combined_texts.append(f"{prompt}\nQ: {q} {h}\nOptions:\n{options_text}\nA: ")

    elif args.task == "gsm8k": 
        if hint:
            answers = [a.split()[-1] for a in examples["answer"]]
            combined_texts = [f"{prompt}\nQ: {q} ({a})\nA: " for q, a in zip(examples["question"], answers)]
        else:
            combined_texts = [f"{prompt}\nQ: {q}\nA: " for q in examples["question"]]

    elif args.task == "svamp":
        if hint:
            combined_texts = [f"{prompt}\nQ: {q} ({a})\nA: " for q, a in zip(examples["question_concat"], examples["answer"])]
        else:
            combined_texts = [f"{prompt}\nQ: {q}\nA: " for q in examples["question_concat"]]

    elif args.task == "cladder":
        if hint:
            combined_texts = [f"{prompt}\nQ: {q} ({a})\nA: " for q, a in zip(examples["question"], examples["answer"])]
        else:
            combined_texts = [f"{prompt}\nQ: {q}\nA: " for q in examples["question"]]

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


def get_majority_vote(predictions_list, datas, args):
    correct, total = 0, 0
    detailed_logs = []

    try:
        for idx, (sample_predictions, data) in enumerate(zip(zip(*predictions_list), datas)):
            sample_log = {
                "index": idx,
                "question": data.get("question", ""),
                "reference_answer": data.get("answer", ""),
                "predictions": [],
                "voting_results": {},
                "final_answer": None,
                "correct": False
            }

            try:
                ref = data.get("answer")
                if ref is None:
                    print(f"Warning: Missing reference answer for index {idx}")
                    continue

                vote_dict = {}
                cleaned_predictions = []

                for pred_idx, pred in enumerate(sample_predictions):
                    try:
                        q_start_idx = pred.find("Q: ")
                        if q_start_idx != -1:
                            pred = pred[:q_start_idx]

                        pred_answer = None
                        pred_log = {
                            "sample_idx": pred_idx,
                            "raw_prediction": pred,
                            "extracted_answer": None,
                            "valid": False
                        }

                        if args.task == "arc_challenge":
                            matches = list(re.findall(r"\b(A|B|C|D)\b", pred))
                            pred_answer = matches[-1] if matches else None

                        elif args.task == "anli_r1":
                            matches = list(re.findall(r"\b(0|1|2)\b", pred))
                            pred_answer = matches[-1] if matches else None

                        elif args.task == "cladder":
                            matches = re.findall(r"\b(yes|no)\b", pred, re.IGNORECASE)
                            pred_answer = matches[-1].lower() if matches else None

                        elif args.task == "cqa":
                            matches = list(re.findall(r"\b(A|B|C|D|E)\b", pred))
                            pred_answer = matches[-1] if matches else None

                        else: 
                            matches = re.findall(r"-?\d+\.?\d*", pred)
                            pred_answer = matches[-1] if matches else None

                        pred_log["extracted_answer"] = pred_answer
                        if pred_answer is not None:
                            pred_log["valid"] = True
                            vote_dict[str(pred_answer)] = vote_dict.get(str(pred_answer), 0) + 1
                            cleaned_predictions.append(pred_answer)

                        sample_log["predictions"].append(pred_log)

                    except Exception as e:
                        print(f"Warning: Error processing prediction {pred_idx} in sample {idx}: {e}")
                        continue

                sample_log["voting_results"] = vote_dict

                if vote_dict:
                    try:
                        max_votes = max(vote_dict.values())
                        top_answers = [ans for ans, votes in vote_dict.items() if votes == max_votes]

                        if len(top_answers) > 1:
                            final_answer = top_answers[0]
                            sample_log["tie_occurred"] = True
                            sample_log["tied_answers"] = top_answers
                        else:
                            final_answer = top_answers[0]

                        sample_log["final_answer"] = final_answer

                        is_correct = False
                        if args.task == "arc_challenge":
                            is_correct = final_answer == ref
                        elif args.task == "cqa":
                            is_correct = final_answer == ref
                        elif args.task == "anli_r1":
                            is_correct = final_answer == str(ref)
                        elif args.task == "cladder":
                            is_correct = final_answer.lower() == str(ref).lower()
                        elif args.task == "svamp":
                            ref_match = re.search(r"-?\d+\.?\d*", str(ref))
                            ref_answer = ref_match.group(0) if ref_match else None
                            is_correct = final_answer.strip() == str(ref_answer).strip()
                        else:
                            ref_match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(ref))
                            ref_answer = ref_match.group(1).strip() if ref_match else None
                            is_correct = final_answer.strip() == str(ref_answer).strip()

                        if is_correct:
                            correct += 1
                            sample_log["correct"] = True

                    except Exception as e:
                        print(f"Warning: Error comparing answers in sample {idx}: {e}")
                        continue

                total += 1
                detailed_logs.append(sample_log)

            except Exception as e:
                print(f"Warning: Error processing sample {idx}: {e}")
                continue

    except Exception as e:
        print(f"Critical error in majority vote: {e}")

    try:
        log_file = f"{args.log_dir}/self_consistency_detailed_log.jsonl"
        with open(log_file, 'a') as f:
            for log in detailed_logs:
                json.dump(log, f)
                f.write('\n')
    except Exception as e:
        print(f"Warning: Failed to write detailed logs: {e}")

    try:
        stats = {
            "total_samples": total,
            "correct_samples": correct,
            "accuracy": correct / total if total > 0 else 0,
            "total_predictions": sum(len(log["predictions"]) for log in detailed_logs),
            "valid_predictions": sum(
                sum(1 for p in log["predictions"] if p["valid"])
                for log in detailed_logs
            ),
            "ties": sum(1 for log in detailed_logs if log.get("tie_occurred", False)),
        }

        stats["valid_prediction_rate"] = (
            stats["valid_predictions"] / stats["total_predictions"]
            if stats["total_predictions"] > 0 else 0
        )

        with open(f"{args.log_dir}/self_consistency_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

    except Exception as e:
        print(f"Warning: Failed to calculate and log statistics: {e}")

    return correct, total

def eval_dev(args, model, rank, test_loader, tokenizer, gen_length, n_shot_prompts):
    generate_fn = model.module.generate if hasattr(model, 'module') else model.generate

    eval_progress_bar = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"Eval [Rank {rank}]",
        position=rank + 1,
        leave=False,
        disable=(rank != 0),
    )

    correctsum, totalsum = 0, 0
    sc_correctsum, sc_totalsum = 0, 0
    
    with torch.no_grad():
        for batch_idx, data in eval_progress_bar:
            try:
                if "input_ids" not in data or "attention_mask" not in data:
                    print(f"Warning: Missing required tensors in batch {batch_idx}")
                    continue

                try:
                    input_ids = data["input_ids"].to(rank)
                    attention_mask = data["attention_mask"].to(rank)
                except Exception as e:
                    print(f"Warning: Error moving tensors to device in batch {batch_idx}: {e}")
                    continue

                try:
                    outputs = generate_fn(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_length,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    generated_tokens = outputs[:, input_ids.size(1):]
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
                    
                    flops_log_file = f"{args.flops_dir}/flops_log_{rank}.json"
                    log_args(flops_log_file, iter=args.exp_iter,idx=batch_idx, split="dev", hint='False' , batch=actual_c, input= actual_c*input_ids.size(1),output= actual_gen_len)
          
                except Exception as e:
                    print(f"Warning: Generation failed for batch {batch_idx}: {e}")
                    continue

                try:
                    generated_tokens = outputs[:, input_ids.shape[-1]:]
                    predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                except Exception as e:
                    print(f"Warning: Decoding failed for batch {batch_idx}: {e}")
                    continue

                all_predictions = [None for _ in range(dist.get_world_size())]
                all_data = [None for _ in range(dist.get_world_size())]

                try:
                    dist.all_gather_object(all_predictions, predictions)
                    dist.all_gather_object(all_data, data)
                except Exception as e:
                    print(f"Warning: All-gather failed for batch {batch_idx}: {e}")
                    continue

                if args.self_consistency > 0:
                    try:
                        sc_predictions_list = []
                        for sc_idx in range(args.self_consistency):
                            try:
                                outputs = generate_fn(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=gen_length,
                                    pad_token_id=tokenizer.eos_token_id,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_k=40
                                )
                                generated_tokens = outputs[:, input_ids.shape[-1]:]
                                sc_predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                                sc_predictions_list.append(sc_predictions)
                            except Exception as e:
                                print(
                                    f"Warning: Self-consistency generation {sc_idx} failed for batch {batch_idx}: {e}")
                                continue

                        sc_all_predictions_list = [None for _ in range(dist.get_world_size())]
                        try:
                            dist.all_gather_object(sc_all_predictions_list, sc_predictions_list)
                        except Exception as e:
                            print(f"Warning: Self-consistency all-gather failed for batch {batch_idx}: {e}")
                            continue
                    except Exception as e:
                        print(f"Warning: Self-consistency evaluation failed for batch {batch_idx}: {e}")
                        continue

                if rank == 0:
                    try:
                        all_predictions = list(chain.from_iterable(all_predictions))
                        merged_data = []
                        for rank_data in all_data:
                            try:
                                for i in range(len(rank_data["question"])):
                                    single_data = {}
                                    for key in rank_data.keys():
                                        single_data[key] = rank_data[key][i]
                                    merged_data.append(single_data)
                            except Exception as e:
                                print(f"Warning: Error merging data from rank {rank}: {e}")
                                continue

                        wrong_examples, correct, total = test_metric_STaR(args, all_predictions, merged_data,
                                                                          args.target_save, tokenizer, rank)
                        correctsum += correct
                        totalsum += total

                        if args.self_consistency > 0 and sc_predictions_list:
                            try:
                                sc_all_predictions_list = list(chain.from_iterable(sc_all_predictions_list))
                                sc_correct, sc_total = get_majority_vote(sc_all_predictions_list, merged_data, args)
                                sc_correctsum += sc_correct
                                sc_totalsum += sc_total
                            except Exception as e:
                                print(f"Warning: Error processing self-consistency results: {e}")
                    except Exception as e:
                        print(f"Warning: Error processing results on rank 0: {e}")
                        continue

                dist.barrier()

            except Exception as e:
                print(f"Warning: Failed to process batch {batch_idx}: {e}")
                continue
             
    if rank == 0:
        try:
            if totalsum > 0:
                accuracy = correctsum / totalsum
                print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
            else:
                print("Warning: No valid examples were processed in base evaluation")

            if args.self_consistency > 0:
                if sc_totalsum > 0:
                    sc_accuracy = sc_correctsum / sc_totalsum
                    print(
                        f"Self-consistency ({args.self_consistency} samples) Evaluation Accuracy: {sc_accuracy * 100:.2f}%")
                else:
                    print("Warning: No valid examples were processed in self-consistency evaluation")
                    sc_accuracy = None

            try:
                log_args(f"{args.log_dir}/eval_log.json",
                         iter=args.exp_iter,
                         split=args.split,
                         accuracy=accuracy if totalsum > 0 else None,
                         sc_accuracy=sc_accuracy if args.self_consistency > 0 and sc_totalsum > 0 else None)
            except Exception as e:
                print(f"Warning: Failed to log evaluation results: {e}")

        except Exception as e:
            print(f"Critical error in final evaluation reporting: {e}")

    return correctsum, totalsum


def clean_existing_eval_logs(args, rank):
    if rank != 0:  
        return

    try:
        if os.path.exists(f"{args.log_dir}/eval_log.json"):
            with open(f"{args.log_dir}/eval_log.json", 'r') as f:
                eval_logs = json.load(f)

            if isinstance(eval_logs, list):
                eval_logs = [log for log in eval_logs
                             if not (log.get('iter') == args.exp_iter and
                                     log.get('split') == 'train')]
            elif isinstance(eval_logs, dict):
                if str(args.exp_iter) in eval_logs:
                    if 'train' in eval_logs[str(args.exp_iter)]:
                        del eval_logs[str(args.exp_iter)]['train']

            with open(f"{args.log_dir}/eval_log.json", 'w') as f:
                json.dump(eval_logs, f, indent=2)
    except Exception as e:
        print(f"Warning: Error cleaning eval_log.json: {e}")

    if args.split == "train":
        for file_path in [f"{args.log_dir}/self_consistency_detailed_log.jsonl"]:
            if os.path.exists(file_path):
                try:
                    temp_file = file_path + '.temp'
                    with open(file_path, 'r') as infile, open(temp_file, 'w') as outfile:
                        for line in infile:
                            try:
                                entry = json.loads(line)
                                if not (entry.get('iter', entry.get('iteration')) == args.exp_iter and
                                        entry.get('split', 'train') == 'train'):
                                    outfile.write(line)
                            except json.JSONDecodeError:
                                continue

                    os.replace(temp_file, file_path)
                except Exception as e:
                    print(f"Warning: Error cleaning {file_path}: {e}")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

        try:
            corr_idx_file = f"{args.idx_save}/{args.split}_corr_idx_{args.exp_iter}.txt"
            if os.path.exists(corr_idx_file):
                os.remove(corr_idx_file)

            if os.path.exists(f"{args.target_save}/correct_data.txt"):
                os.remove(f"{args.target_save}/correct_data.txt")
        except Exception as e:
            print(f"Warning: Error cleaning generated data files: {e}")

def evaluate(args, model, rank, world_size, test_loader, tokenizer, gen_length, target_save, n_shot_prompts, n_shot_prompts_hint):
    dist.barrier()  
    if args.split == "train":  
        clean_existing_eval_logs(args, rank)
    dist.barrier() 

    model.eval()

    if args.split == "train": 
        wrong_datasets, correctsum, totalsum = eval_examples(args, model, rank, test_loader, tokenizer, gen_length, n_shot_prompts, hint=False)
        wrong_datasets = broadcast_list(wrong_datasets, src_rank=0)
        wrong_test_loader, sampler_wrong_test = get_wrong_examples_dataloader_STaR(args, wrong_datasets, rank, world_size)
        if args.no_hint:
            wrong_datasets, correctsum2, totalsum2 = eval_examples(args, model, rank, wrong_test_loader, tokenizer, gen_length, n_shot_prompts, hint=False)
            correctsum += correctsum2
            correctsum_hint, totalsum_hint = "_", "_"
        else:
            wrong_wrong_datasets, correctsum_hint, totalsum_hint = eval_examples(args, model, rank, wrong_test_loader, tokenizer, gen_length, n_shot_prompts_hint, hint=True)

    else:
        correctsum, totalsum = eval_dev(args, model, rank, test_loader, tokenizer, gen_length, n_shot_prompts)
        correctsum_hint, totalsum_hint = "_", "_"
    dist.barrier()

    if rank == 0:
        merge_flops_logs(args)
    return correctsum, totalsum, correctsum_hint, totalsum_hint

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
    parser.add_argument("--flops_dir", type=str, default="", help="logging dir")


    return parser.parse_args()

def fsdp_main(rank, world_size, args):
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

    # Load ckpt
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

    args.batch_size = args.test_batch_size 

    train_loader, sampler_train, test_loader, sampler_test = get_dataloader(args, tokenizer, rank, world_size)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    if args.split == "train":
        init_start_event.record()
        correct, total, correct_hint, total_hint = evaluate(args, model, rank, world_size, train_loader, tokenizer, args.gen_length, args.target_save, n_shot_prompts, n_shot_prompts_hint)
        init_end_event.record()
        if rank == 0:
            log_args(f"{args.log_dir}/elapsed_time_log.json", iter=args.exp_iter, log_point="gen_train_data", time=init_start_event.elapsed_time(init_end_event) / 1000)

    elif args.split == "dev":
        init_start_event.record()
        correct, total, correct_hint, total_hint = evaluate(args, model, rank, world_size, test_loader, tokenizer, args.gen_length, args.target_save, n_shot_prompts, n_shot_prompts_hint)
        init_end_event.record()
        if rank == 0:
            log_args(f"{args.log_dir}/elapsed_time_log.json", iter=args.exp_iter, log_point="test", time=init_start_event.elapsed_time(init_end_event) / 1000)

    if rank == 0:
        accuracy = correct / total
        if args.split == "train":
            if not args.no_hint:
                hint_accuracy = correct_hint / total_hint
            else:
                hint_accuracy = "_"
            print(f"{args.split}, {args.task}, accuracy: {accuracy}, hint_accuracy: {hint_accuracy}")
            log_args(f"{args.log_dir}/eval_log.json", iter=args.exp_iter, split=args.split, accuracy=accuracy, hint_accuracy=hint_accuracy)

    dist.barrier()
    cleanup()

if __name__ == "__main__":

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
    args.accumulate = params["accumulate"]
    args.lora = params.get("lora", None)
    args.inference_temp = params["inference_temp"]
    args.no_hint = params["no_hint"]
    args.base_model_path = params["base_model_path"]

    args.name = params["name"]
    args.idx_save = params["target_save"] 
    args.target_save = params["target_save"] if split != "dev" else f'{args.task}/new_dev.txt'
    args.model_dir = params["model_dir"]
    args.total_steps = params.get("total_steps", 0)
    
    args.method = params["method"]

    torch.manual_seed(args.seed)

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)



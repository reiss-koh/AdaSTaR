import argparse
import os
import re
import pickle
from pathlib import Path
from typing import List
import json
import glob
import ftfy
from lm_dataformat import Reader
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from utils import build_problem_id_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="""Converts a text dataset into PyTorch-compatible format.""")
    parser.add_argument("input_path", type=str, help="Path to an input file or a directory containing input files.")
    parser.add_argument("name", type=str, help="Name of output file will be {name}.pt.")
    parser.add_argument("--idx_save", type=str, help="Directory containing index files")
    parser.add_argument("--split", type=str, default="train", help="Data split (train/val/test)")
    parser.add_argument("--exp_iter", type=int, default=0, help="Experiment iteration number")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory (default: current directory).")
    parser.add_argument("--model_name", type=str, help="Model name for tokenizer")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length (default: 512).")

    parser.add_argument("--task", type=str, default="", help="Task name for problem mapping")
    parser.add_argument("--experiment_name", type=str, default="", help="Experiment name for finding stats files")
    parser.add_argument("--log_dir", type=str, help="Directory containing problem tracking stats")
    parser.add_argument("--problem_tracking_file", type=str, help="problem tracking stats")

    cleaning_args = parser.add_argument_group('data cleaning arguments')
    cleaning_args.add_argument("--normalize-with-ftfy", action="store_true", help="Normalize text with ftfy")
    cleaning_args.add_argument("--normalize-with-wikitext-detokenize", action="store_true",
                               help="Use wikitext detokenizer")
    cleaning_args.add_argument("--min-unique-tokens", type=int, default=0,
                               help="Exclude documents with fewer unique tokens.")
    cleaning_args.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")

    args = parser.parse_args()
    args.input_path = Path(args.input_path)
    return args


def normalize_text(text):
    return ' '.join(text.strip().split()).lower()


def get_files(input_path: Path) -> List[str]:
    supported_file_types = ["jsonl.zst", "correct_data.txt", ".xz", ".tar.gz"]
    if input_path.is_dir():
        files = [list(Path(input_path).glob(f"*{ft}")) for ft in supported_file_types]
        files = [f for sublist in files for f in sublist]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert any(str(input_path).endswith(ft) for ft in supported_file_types), f"Unsupported file type: {input_path}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path=}")
    return [str(f) for f in files]


def wikitext_detokenizer(string):
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def save_to_pt(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def arrays_to_sequences(token_list_iterable, sequence_length=1024):
    accum = []
    for l in token_list_iterable:
        accum.extend(l)
        while len(accum) >= sequence_length:
            yield accum[:sequence_length]
            accum = accum[sequence_length:]
    if accum:
        yield accum


def create_pytorch_dataset(files, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    eos_token_id = tokenizer.eos_token_id

    indices = {}
    if args.idx_save:
        idx_file = os.path.join(args.idx_save, f"{args.split}_corr_idx_{args.exp_iter}.txt")
        print(f"Reading index file: {idx_file}")
        try:
            with open(idx_file, 'r') as f:
                current_idx = None
                for line in f:
                    line = line.strip()
                    if line.startswith('idx: '):
                        current_idx = int(line.split('idx: ')[1])
                    elif line.startswith('Q: '):
                        current_question = normalize_text(line[3:])
                        indices[current_question] = current_idx
            print(f"Loaded {len(indices)} question-index pairs")
        except Exception as e:
            print(f"Error reading index file: {e}")
            raise

    wrong_sequences = []
    normal_sequences = []

    max_token_len = 0
    num_skipped = 0

    consistently_wrong_problems = {}
    wrong_problems_added = 0

    if args.include:
        log_dir = args.log_dir if hasattr(args, 'log_dir') and args.log_dir else f"{args.task}/{args.experiment_name}"
        stats_files = glob.glob(os.path.join(log_dir, "problem_tracking_stats_iter*.json"))

        if stats_files:
            latest_stats_file = sorted(stats_files,
                                       key=lambda x: int(re.search(r'iter(\d+)', x).group(1)))[-1]

            print(f"Loading consistently wrong problems from: {latest_stats_file}")

            try:
                with open(latest_stats_file, 'r') as f:
                    stats_data = json.load(f)

                if "not_consistently_solved_ids" in stats_data:
                    wrong_problem_ids = stats_data["not_consistently_solved_ids"]
                    print(f"Found {len(wrong_problem_ids)} consistently wrong problems")

                    if hasattr(args, 'task') and args.task:
                        problem_mapping = build_problem_id_mapping(args, args.task, tokenizer)

                        for problem_id in wrong_problem_ids:
                            if problem_id in problem_mapping:
                                consistently_wrong_problems[problem_id] = problem_mapping[problem_id]

                        print(f"Loaded details for {len(consistently_wrong_problems)} consistently wrong problems")
            except Exception as e:
                print(f"Error loading consistently wrong problems: {e}")
                print("Continuing without consistently wrong problems")

        if consistently_wrong_problems:
            print("Adding consistently wrong problems to the dataset...")

            for problem_id, problem_data in consistently_wrong_problems.items():
                question = problem_data.get("question", "")
                answer = problem_data.get("answer", "")

                if question and answer:
                    qa_text = f"Q: {question}\nA: {answer}"

                    tokens = tokenizer.encode(qa_text, truncation=True, max_length=args.max_length)

                    if len(tokens) <= args.max_length:
                        wrong_sequences.append({
                            'tokens': tokens,
                            'idx': int(problem_id) if problem_id.isdigit() else -1,
                            'question': question,
                            'is_wrong_problem': True 
                        })
                        wrong_problems_added += 1
                        max_token_len = max(max_token_len, len(tokens))
                    else:
                        print(f"Warning: Wrong problem ID {problem_id} skipped due to length")

            print(f"Added {wrong_problems_added} consistently wrong problems to the dataset")

    for file in tqdm(files, desc="Processing files"):
        reader = Reader(file)
        for doc in reader.stream_data():
            if args.normalize_with_ftfy:
                doc = ftfy.fix_text(doc, normalization='NFKC')
            if args.normalize_with_wikitext_detokenize:
                doc = wikitext_detokenizer(doc)

            if args.idx_save:
                qa_pairs = doc.split('\n\n')
                for qa in qa_pairs:
                    if not qa.strip():
                        continue

                    try:
                        question_parts = qa.split('\nA:')[0].split('Q: ')
                        if len(question_parts) > 1:
                            question = question_parts[1].split('\n')[0]
                            question = normalize_text(question)
                            idx = indices.get(question, -1)
                            if idx == -1:
                                print(f"Warning: No index found for question: {question}")
                                continue
                        else:
                            continue

                        tokens = tokenizer.encode(qa, truncation=True, max_length=args.max_length)
                        max_token_len = max(max_token_len, len(tokens))

                        if len(tokens) <= args.max_length:
                            normal_sequences.append({
                                'tokens': tokens,
                                'idx': idx,
                                'question': question
                            })
                        else:
                            num_skipped += 1

                    except Exception as e:
                        print(f"Error processing QA pair: {e}")
                        print(f"QA pair content: {qa[:100]}...")
                        continue
            else:
                split_docs = doc.split(tokenizer.eos_token)
                for sub_doc in split_docs:
                    sub_doc = sub_doc.strip()
                    if sub_doc:
                        tokens = tokenizer.encode(sub_doc) + [eos_token_id]
                        max_token_len = max(max_token_len, len(tokens))
                        if len(tokens) <= args.max_length:
                            normal_sequences.append(tokens)
                        else:
                            num_skipped += 1

    all_sequences = wrong_sequences + normal_sequences

    print(f"Number of wrong sequences: {len(wrong_sequences)}")
    print(f"Number of normal sequences: {len(normal_sequences)}")
    print(f"Total number of sequences: {len(all_sequences)}")
    print(f"Number of sequences skipped due to length: {num_skipped}")
    print(f"Maximum token length: {max_token_len}")

    if len(all_sequences) == 0:
        raise ValueError("No sequences were created. Please check the input data and parameters.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, f"{args.name}.pt")
    with open(output_file, 'wb') as f:
        pickle.dump(all_sequences, f)
    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    args = parse_args()
    args.include = True
    torch.manual_seed(args.seed)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    files = get_files(args.input_path)
    create_pytorch_dataset(files, args)
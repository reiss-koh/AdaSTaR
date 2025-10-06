import json
import numpy as np
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# File paths
file_paths = [
    "train.jsonl",
    "test.jsonl"
]


def compute_token_statistics(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if not lines:
                print(f"File {file_path} is empty.")
                return

            token_lengths = []

            for line in lines:
                try:
                    # Parse the JSON object
                    json_obj = json.loads(line)

                    # Combine values of 'question' and 'answer'
                    q = json_obj.get("question_concat", "")
                    # t = json_obj.get("Body", "")
                    f = json_obj.get("Equation", "")
                    a = json_obj.get("Answer", "")
                    combined_text = f"Q: {q}\nA: {f}\n####{a}"

                    # Compute token length
                    tokens = tokenizer(combined_text, return_tensors="pt", truncation=False).input_ids
                    token_lengths.append(tokens.shape[1])

                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file_path}")
                    continue

            # Calculate statistics
            if token_lengths:
                mean_tokens = np.mean(token_lengths)
                std_dev_tokens = np.std(token_lengths)
                max_tokens = np.max(token_lengths)

                print(f"\nFile: {file_path}")
                print("-" * 40)
                print(f"Mean token count: {mean_tokens:.2f}")
                print(f"Standard deviation: {std_dev_tokens:.2f}")
                print(f"Max token count: {max_tokens}")
            else:
                print(f"No valid data in {file_path}.")

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred with file {file_path}: {e}")


# Process each file
for file_path in file_paths:
    compute_token_statistics(file_path)

import heapq
import pickle
import json
from datasets import load_from_disk, load_dataset, concatenate_datasets, Dataset
import pickle
from safetensors.torch import load_file
import copy

from utils import preprocess_function, add_idx_to_batch

class MinHeap:
    def __init__(self):
        self._heap = []

    def push(self, priority1, priority2, item):
        heapq.heappush(self._heap, (priority1, priority2, item))

    def pop(self):
        if not self._heap:
            return None
        priority1, priority2, item = heapq.heappop(self._heap)
        return (priority1, priority2, item)

    def peek(self):
        if not self._heap:
            return None
        priority1, priority2, item = self._heap[0]
        return (priority1, priority2, item)

    def __len__(self):
        return len(self._heap)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._heap, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            heap_data = pickle.load(f)
        
        instance = cls()
        instance._heap = heap_data
        return instance

def get_dataloader_adastar(args, tokenizer, rank, world_size):
    dataset, dataset_train, dataset_test = None, None, None
    if args.task == "gsm8k":
        dataset = load_from_disk("../datasets/data_gsm8k")
        dataset_test = load_dataset("json", data_files="./datasets/data_gsm8k/test.jsonl")["train"]

        dataset_train = dataset["train"].map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
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
    
    elif args.task == "svamp":
        dataset_train = load_dataset("json", data_files="./datasets/data_svamp/train.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_svamp/test.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "cladder":
        dataset_train = load_dataset("json", data_files="./datasets/data_cladder_split/cladder_train_long.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_cladder_split/cladder_test.jsonl")["train"]
        
        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    elif args.task == "anli_r1":
        dataset_train = load_dataset("json", data_files="./datasets/data_anli/train_r1_modified.jsonl")["train"]
        dataset_test = load_dataset("json", data_files="./datasets/data_anli/test_r1_modified.jsonl")["train"]

        dataset_train = dataset_train.map(lambda examples: preprocess_function(args, examples, tokenizer, "train"), batched=True)
        dataset_test = dataset_test.map(lambda examples: preprocess_function(args, examples, tokenizer, "test"), batched=True)

    dataset_train = dataset_train.map(add_idx_to_batch, with_indices=True, batched=True)

    if args.exp_iter == 1:
        if args.method in {"adastar_new_square", "adastar_new"}:
            winlose = initialize_winlose_new(args, args.log_dir + "/winlose.json", dataset_train, rank)
            heap = initialize_heap_new(args, args.log_dir + "/heap.pkl", winlose, rank)
    else:
        if args.method in {"adastar_new_square", "adastar_new"}:
            winlose = json.load(open(args.log_dir + "/winlose.json", "r"))
            heap = MinHeap.load(args.log_dir + "/heap.pkl")

    custom_order = heap_to_custom_order(args, heap)

    dataset_train_ordered = dataset_train.select(custom_order)

    return dataset_train_ordered, winlose, heap

def heap_to_custom_order(args, heap):
    custom_order = []
    heap_copy = copy.deepcopy(heap)

    while len(heap_copy) > 0:
        if args.method in {"adastar_new_square", "adastar_new"}:
            _, _, item = heap_copy.pop()
        itemidx = int(item.split("_")[1])
        custom_order.append(itemidx)
    return custom_order

def initialize_winlose_new(args, output_file, dataset, rank):
    winlose = {}
    for i in range(len(dataset)):
        winlose[f"id_{i}"] = {"iter": 0, "win": 0, "total": 0}
    if rank == 0:
        json.dump(winlose, open(output_file, "w"))
    return winlose

def initialize_heap_new(args, output_file, winlose, rank):
    heap = MinHeap()
    for key in winlose:
        if winlose[key]["total"] == 0:
            heap.push(winlose[key]["iter"], 0, key)
        else:
            heap.push(winlose[key]["iter"], winlose[key]["win"]/winlose[key]["total"], key)
    if rank == 0:
        heap.save(output_file)
    return heap

def update_winlose_new(prev_winlose, new_winlose, num_used_data):
     cnt = 0
     for key in new_winlose:
         prev_winlose[key]["iter"] = new_winlose[key]["iter"]
         prev_winlose[key]["win"] = new_winlose[key]["win"]
         prev_winlose[key]["total"] = new_winlose[key]["total"]
         cnt += 1
         if cnt >= num_used_data:
             break
     return prev_winlose

def update_minheap_new(updated_winlose, heap, num_used_data):
     for i in range(num_used_data):
         _, _, item = heap.pop()
         if updated_winlose[item]["total"] == 0:
             heap.push(updated_winlose[item]["iter"], 0, item)
         else:
             heap.push(updated_winlose[item]["iter"], updated_winlose[item]["win"]/updated_winlose[item]["total"], item)
     return heap

def update_minheap_winlose_front(heap, prev_winlose, new_winlose, num_pop_data):
    for i in range(num_pop_data):
        p1, p2, item = heap.pop()

        heap.push(new_winlose[item]["iter"], new_winlose[item]["win"]/new_winlose[item]["total"], item)

        prev_winlose[item]["iter"] = new_winlose[item]["iter"]
        prev_winlose[item]["win"] = new_winlose[item]["win"]
        prev_winlose[item]["total"] = new_winlose[item]["total"]

    return heap, prev_winlose


def log_stable_minheap_new(heap, target_save):
    winrate_order = []
    heap_copy = copy.deepcopy(heap)
    while len(heap_copy) > 0:
        priority1, priority2, item = heap_copy.pop()
        winrate_order.append({"iter": priority1, "winrate": priority2, "id": item})
    json.dump(winrate_order, open(target_save, "w"))
    

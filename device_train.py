import os
import argparse
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

import json
from tqdm import tqdm

from utils import get_model_tokenizer, get_optimizer_scheduler_step_based, setup, cleanup, log_args, \
    get_dataloader_STaR, get_loaded_model_tokenizer, merge_flops_logs


def create_labels(input_ids, rank, tokenizer):
    batch_size, seq_len = input_ids.shape
    labels = input_ids.clone().to(torch.int64)

    a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    colon_token_id = tokenizer.encode(":", add_special_tokens=False)[0]

    for i in range(batch_size):
        last_a_colon_position = -1
        for j in range(seq_len - 1):
            if input_ids[i, j] == a_token_id and input_ids[i, j + 1] == colon_token_id:
                last_a_colon_position = j

        if last_a_colon_position != -1:
            labels[i, :last_a_colon_position + 2] = -100 
        else:
            labels[i, :] = -100 
    return labels


def gather_and_merge_logs(tensor, rank, world_size):
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return torch.stack(gather_list)


def train_step_based(args, model, tokenizer, rank, world_size, train_loader, optimizer, scheduler, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    global_step = 0
    total_steps = args.total_steps

    effective_batch_size = args.batch_size * args.grad_accum * world_size
    args.required_data_num = total_steps * effective_batch_size

    used_indices = set()
    step_indices = {}

    dataloader_iterator = iter(train_loader)
    progress_bar = tqdm(
        total=total_steps,
        desc=f"Training [Rank {rank}]",
        position=rank,
        leave=False,
        disable=(rank != 0),
    )

    args.log_interval = max(1, round(total_steps // args.log_divisor))

    while global_step < total_steps:
        step_batch_indices = []

        for idx in range(args.grad_accum):
            try:
                data = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_loader)
                if sampler:
                    sampler.set_epoch(global_step // len(train_loader))
                data = next(dataloader_iterator)

            batch_indices = data["idx"]
            step_batch_indices.extend(batch_indices)

            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = create_labels(input_ids, rank, tokenizer)

            input_ids, attention_mask, labels = (
                input_ids.to(rank),
                attention_mask.to(rank),
                labels.to(rank),
            )

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss / args.grad_accum
            loss.backward()

            flops_log_file = f"{args.log_dir}/flops_log_{rank}.json"
            log_args(flops_log_file, iter=args.exp_iter, idx=global_step, split="train",
                    batch=args.batch_size, grad_accum=args.grad_accum,
                    context_len=args.batch_size*input_ids.size(1))
            
            ddp_loss[0] += loss.item() * args.grad_accum
            ddp_loss[1] += len(input_ids)

        step_batch_indices = [int(idx) if isinstance(idx, torch.Tensor) else idx for idx in step_batch_indices]
        indices_tensor = torch.tensor(step_batch_indices, device=rank)
        gathered_indices = [torch.zeros_like(indices_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_indices, indices_tensor)

        all_step_indices = []
        for gathered in gathered_indices:
            all_step_indices.extend(gathered.cpu().tolist())

        used_indices.update(all_step_indices)
        step_indices[global_step] = all_step_indices

        if rank == 0 and ((global_step + 1) % args.log_interval == 0 or (global_step + 1) == total_steps):
            log_entry = {
                "iter": args.exp_iter,
                "step": global_step,
                "batch_indices": all_step_indices,
                "total_unique_indices": len(used_indices),
                "batch_size": len(all_step_indices)
            }

            os.makedirs(args.log_dir, exist_ok=True)
            index_log_file = f"{args.log_dir}/training_indices_log.json"

            try:
                with open(index_log_file, 'r') as f:
                    logs = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logs = []

            logs.append(log_entry)
            with open(index_log_file, 'w') as f:
                json.dump(logs, f, indent=2)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        if (global_step + 1) == total_steps:
            save_consolidated_model(args, model, args.total_steps, args.model_dir, rank)

        if (global_step + 1) % args.log_interval == 0 or (global_step + 1) == total_steps:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            avg_loss = (ddp_loss[0] / ddp_loss[1]).item()

            if rank == 0:
                progress_bar.set_postfix({
                    "Loss": avg_loss,
                    "LR": current_lr,
                    "Unique Indices": len(used_indices)
                })

                trainfile = f"{args.log_dir}/train_log.json"
                log_entry = {
                    "iter": args.exp_iter,
                    "step": global_step,
                    "loss": avg_loss,
                    "learning_rate": current_lr,
                    "unique_indices": len(used_indices)
                }

                try:
                    with open(trainfile, 'r') as f:
                        train_logs = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    train_logs = []

                train_logs.append(log_entry)
                with open(trainfile, 'w') as f:
                    json.dump(train_logs, f, indent=2)

            dist.barrier()
            ddp_loss = torch.zeros(2).to(rank)

        global_step += 1
        progress_bar.update(1)

    if rank == 0:
        final_stats = {
            "iter": args.exp_iter,
            "total_steps": total_steps,
            "total_unique_indices_used": len(used_indices),
            "unique_indices": sorted(list(used_indices)),
            "indices_per_step": {str(k): v for k, v in step_indices.items()}
        }
        with open(f"{args.log_dir}/final_indices_stats.json", 'w') as f:
            json.dump(final_stats, f, indent=2)

    progress_bar.close()


def save_consolidated_model(args, model, step, path, rank):  
    assert path
    if rank == 0:
        step_path = os.path.join(path, f"step_{step}")
        os.makedirs(step_path, exist_ok=True)

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        full_state_dict = model.state_dict()
        if rank == 0:
            if args.lora:  
                lora_state_dict = {k: v for k, v in full_state_dict.items() if "lora_" in k}
                full_state_dict = lora_state_dict

            torch.save(full_state_dict, f"{step_path}/lm.pt")
            print(f"model saved at {step_path}/lm.pt")
            if args.delete_path is not None:
                ckpt_path=args.delete_path
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    print(f"Model file {ckpt_path} deleted successfully.")
                else:
                    print(f"Model file {ckpt_path} not found.")
    dist.barrier()


def fsdp_main(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank) 

    if args.model_path == None: 
        if args.base_model_path != None:
            model, tokenizer = get_loaded_model_tokenizer(args, args.base_model_path, args.model_name, rank)
            if rank == 0:
                print(f"[Train] base model loaded from {args.base_model_path}")
        else:
            model, tokenizer = get_model_tokenizer(args, args.model_name, rank)
            if rank == 0:
                print("[Train] base model path == None, using hf model")
    else:
        model, tokenizer = get_loaded_model_tokenizer(args, args.model_path, args.model_name, rank)
        if rank == 0:
            print(f"[Train] model loaded from {args.model_path}")

    with open(args.data_dir, 'r') as f:
        pt_file_path = f.read().strip() 
        args.dataset_path = pt_file_path
    train_loader, sampler_train = get_dataloader_STaR(args, tokenizer, rank, world_size)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    optimizer, scheduler = get_optimizer_scheduler_step_based(args, model, train_loader)
    init_start_event.record()

    train_step_based(args, model, tokenizer, rank, world_size, train_loader, optimizer, scheduler,
                     sampler=sampler_train)

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        log_args(f"{args.log_dir}/../elapsed_time_log.json", iter=args.exp_iter, log_point="train",
                 time=init_start_event.elapsed_time(init_end_event) / 1000)
        merge_flops_logs(args)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Finetune')
    parser.add_argument("--config", type=str, default="configs/base_fsdp.json", help="Config file location")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling data (default: 10)")

    parser.add_argument("--exp_iter", type=int, default=-1, help="exp iteration for logging")
    parser.add_argument("--log_dir", type=str, default="", help="logging dir")
    parser.add_argument("--flops_dir", type=str, default="", help="logging dir")
    parser.add_argument("--data_dir", type=str, default="", help="data dir")
    parser.add_argument("--model_path", type=str, default=None, help="model_path")
    parser.add_argument("--delete_path", type=str, default=None, help="delete_path")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    params = json.load(open(args.config))

    args.batch_size = params["batch_size"]
    args.test_batch_size = params["test_batch_size"]
    args.grad_accum = params["grad_accumulation"]
    args.gen_length = params["gen_length"]
    args.ckpt_save = params["model_dir"]
    args.log_divisor = params["log_divisor"]
    args.lr = params["lr"]
    args.weight_decay = params["weight_decay"]
    args.warm_up_steps = params["warm_up_steps"] 
    args.scheduler = params["scheduler"]  
    args.optimizer = params["optimizer"] 
    args.precision = params["precision"] 
    args.model_name = params["model_name"]
    args.max_length = params["max_length"]
    args.task = params["task"]
    args.lora = params.get("lora", None)
    args.accumulate = params["accumulate"] 
    args.split = 'ft'
    args.base_model_path = params["base_model_path"]
    
    args.total_steps = params["total_steps"]
    args.name = params["name"]
    args.model_dir = params["model_dir"]
    args.method = params["method"]
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)

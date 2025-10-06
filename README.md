# AdaSTaR: Adaptive Sampling in Training Self-Taught Reasoners

## Contents

- [Project Structure](#Project-Structure)
- [Set up & Installation](#Set-up-&-Installation)
- [Run an experiment](#run-an-experiment)
- [Reference](#reference)


# Project Structure
```
AdaSTaR/
├── configs/                          # Configuration files for AdaSTaR (e.g., hyperparameters, experiment setups)
│
├── configs_method/                   # Configuration files for Adastar
│
├── datasets/                         # Datasets used for this project
│
├── n_shot_prompts/                   # Prompt files for few-shot (N-shot) learning setups
│
├── create_finetune_tfrecords.py      # Script for generating TF Record-format data used in AdaSTaR training
├── device_inference_adastar_new_square.py  # Inference script for AdaSTaR using a modified square-based structure
├── device_inference_adastar_new.py   # Standard inference script for AdaSTaR stochastic version
├── device_inference.py               # General-purpose inference script (specifically for evaluation)
├── device_train.py                   # Main training script for both AdaSTaR and its stochastic version
├── iteration_train.py                # Script for iterative training loop setups
├── README.md                         # Main documentation file
├── requirements.txt                  # List of required Python packages
├── utils_adastar.py                  # Utility functions specific to AdaSTaR
└── utils.py/                         # Utility functions for general STaR flow
```
This repository implements AdaSTaR and its stochastic version, supporting both training and inference workflows.
More detailed instructions on how to set up and run experiments are provided in the sections below.


## Set up & Installation

```bash
# Clone the repository
git clone
# Install dependencies
pip install -r requirements.txt
```


## Run an experiment

### AdaSTaR
```bash
python iteration_train.py --config=configs/example.json --method=adastar_new_square --seed=10
```

### AdaSTaR - Stochastic Version
```bash
python iteration_train.py --config=configs/example.json --method=adastar_new --seed=10
```

Several predefined configuration files are available under the `configs/` directory. Each config file is named after the dataset or model it corresponds to.  
You can select the appropriate config file depending on the dataset or model you want to use.

The naming convention is as follows:

- `{dataset}.json`: uses the **Llama-3.2-3B** model
- `{dataset}_qwen.json`: uses the **Qwen2.5-3B** model
- `{dataset}7b.json`: uses the **Gemma-7B** model

### Available Datasets

The following datasets are currently supported:

- `anli_r1`
- `arc_challenge`
- `cladder`
- `cqa` (CommonsenseQA)
- `gsm8k`
- `svamp`

### Model Support per Dataset

| Dataset       | Llama-3.2-3B | Qwen2.5-3B | Gemma-7B |
|---------------|:------------:|:----------:|:--------:|
| anli_r1       | ✅          | ❌         | ✅      |
| arc_challenge | ✅          | ✅         | ✅      |
| cladder       | ✅          | ❌         | ❌      |
| cqa           | ✅          | ❌         | ❌      |
| gsm8k         | ✅          | ✅         | ❌      |
| svamp         | ✅          | ✅         | ✅      |

✅ **LLaMA 3B** supports **all datasets**.  
✅ **Qwen 3B** supports: `arc_challenge`, `gsm8k`, `svamp`  
✅ **Gemma 7B** supports: `anli_r1`, `arc_challenge`, `svamp`  

❌ in the table simply means that a config file has not been provided yet.  
You can still use that dataset with the model by creating a compatible config file manually.  
Detailed instructions on how to write a config file are provided in the section below.


## Configuration File Structure

Each experiment is configured via a JSON file located in the `configs/` directory.  
These configuration files define the dataset, model type, training parameters, and more.

A typical config file for 3B models (e.g., LLaMA 3B, Qwen 3B) looks like this:

```json
{
  "epochs": 1,
  "grad_accumulation": 1,
  "gen_length": 381,
  "batch_size": 2,
  "test_batch_size": 32,
  "lr": 1e-05,
  "weight_decay": 0.01,
  "warm_up_steps": 100,
  "model_dir": "checkpoints/",
  "log_divisor": 100,
  "save_divisor": 5,
  "exp_name": "testrun",
  "optimizer": "Adam",
  "scheduler": "linear",
  "precision": "bf16",
  "model_name": "meta-llama/Llama-3.2-3B",
  "max_length": 512,
  "n_shot": 6,
  "self_consistency": 0,
  "delete_model_after_loading": true,
  "accumulate": false,
  "task": "arc_challenge",
  "inference_temp": 1.0,
  "no_hint": false,
  "base_model_path" : null
}
```

For 7B models (e.g., Gemma 7B), an additional lora section is required:
```json
{
  ...
  "lora": {
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1
  }
}
```

If you're creating a new config for a model-dataset combination not included by default,
use one of the existing files as a template and modify it accordingly.

## Log File Storage

Logs are saved inside the corresponding method's directory using the following structure:
```
{dataset_name}/{experiment_name}/eval_log.json
```

- `{dataset_name}`: the name of the dataset used  
- `{experiment_name}`: a unique identifier in the format:  
  `{dataset_name}_{method_name}_{seed}`

#### Example

If you're using the `gsm8k` dataset with the `adastar_new_square` method and seed `10`, the log file will be saved at:
```
gsm8k/gsm8k_adastar_new_square_10/eval_log.json
```
These logs contain evaluation results, and are useful for comparing performance across different methods, datasets.

## Reference

If you find this work helpful, please consider citing our paper:
```bibtex
@inproceedings{koh2025adastar,
  title={AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners},
  author={Koh, Woosung and Oh, Wonbeen and Jang, Jaein and Lee, MinHyung and Kim, Hyeongjin and Kim, Ah Yeon and Kim, Joonkee and Lee, Junghyun and Kim, Taehyeon and Yun, Se-Young},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=D6PwC6Xogv}
}
```

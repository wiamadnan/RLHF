"""
python reward_modeling.py \
    --model_name_or_path=distilbert/distilbert-base-uncased\
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --max_length=512
"""

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config

from utils import print_config

# Ensure progress bars are handled correctly in environments like Jupyter Notebooks
tqdm.pandas()


# Parse command line arguments into dataclasses for easy access
parser = HfArgumentParser((RewardConfig, ModelConfig))
reward_config, model_config = parser.parse_args_into_dataclasses()
reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

# Print configurations
print_config(reward_config, "Reward Configuration")
print_config(model_config, "Model Configuration")

#-------- Model & Tokenizer --------#
torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name_or_path, num_labels=1, **model_kwargs
)

#-------- Dataset --------#
raw_datasets = load_dataset("Anthropic/hh-rlhf")

# Tokenize chosen/rejected pairs of inputs
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

# Preprocess the dataset and filter out examples that are longer than args.max_length
train_dataset = raw_datasets["train"]

# Reduce the size of the dataset to 10%
small_train_dataset = train_dataset.shuffle(seed=42).select(
    range(int(0.1 * len(train_dataset)))
)

# Preprocess the training dataset
small_train_dataset = small_train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)

# Filter out examples that are longer than reward_config.max_length for the training dataset
small_train_dataset = small_train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
    and len(x["input_ids_rejected"]) <= reward_config.max_length
)

# Similarly, preprocess the validation dataset
eval_dataset = raw_datasets["test"].map(
    preprocess_function,
    batched=True,
    num_proc=4,
)

# Filter out examples that are longer than reward_config.max_length for the validation dataset
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
    and len(x["input_ids_rejected"]) <= reward_config.max_length
)

#-------- Training --------#
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=small_train_dataset,
    eval_dataset=eval_dataset,
    peft_config=get_peft_config(model_config),
)
trainer.train()
trainer.save_model(reward_config.output_dir)
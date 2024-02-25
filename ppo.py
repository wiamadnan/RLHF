"""
python ppo.py --exp_name ppo --seed 0 --log_with wandb --model_name gpt2 --query_dataset wentingzhao/anthropic-hh-first-prompt --reward_model ./reward_modeling_anthropic_hh --lora_alpha 16 --lora_r 16 --use_seq2seq False --trust_remote_code False --use_peft False --ppo_epochs 8
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForSequenceClassification

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

from utils import print_config, ScriptArguments

# Ensure progress bars are handled correctly in environments like Jupyter Notebooks
tqdm.pandas()

# Parse command line arguments into dataclasses for easy access
parser = HfArgumentParser((ScriptArguments, PPOConfig))
args, ppo_config = parser.parse_args_into_dataclasses()

# Print configurations
print_config(ppo_config, "PPO Configuration")
print_config(args, "Model Configuration")

# We then define the arguments to pass to the reward pipeline.
# We set `return_all_scores` to True to get the reward for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

#-------- Dataset --------#
def build_dataset(config, input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("wentingzhao/anthropic-hh-first-prompt", split="train")
    ds = ds.rename_column("user", "query")
    ds = ds.remove_columns(["source", "system"])
    ds = ds.filter(lambda x: len(x["query"]) > 200, batched=False)
    
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(ppo_config)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)


#-------- Model & Tokenizer --------#
trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead

if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
    device_map = None
    peft_config = None
else:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    ref_model = None
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.eos_token_id

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# We then build the reward pipeline, passing the model name and the
# rewardine arguments. 
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
reward_model = AutoModelForSequenceClassification.from_pretrained(ppo_config.reward_model)
reward_tokenizer = AutoTokenizer.from_pretrained(ppo_config.reward_model)

if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        classif_pipe = pipeline("text-classification", model=reward_model, tokenizer=reward_tokenizer, device=device)
else:
    classif_pipe = pipeline("text-classification", model=reward_model, tokenizer=reward_tokenizer, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if classif_pipe.tokenizer.pad_token_id is None:
    classif_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if classif_pipe.model.config.pad_token_id is None:
    classif_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
}

#-------- Training --------#
for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    print(_epoch)
    query_tensors = batch["input_ids"]

    # Get response from gpt2
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = classif_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = classif_pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output[0]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

model.save_pretrained('./policy_anthropic_hh/model')
tokenizer.save_pretrained('./policy_anthropic_hh/tokenizer')


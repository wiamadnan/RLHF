# Deep RLHF (Reinforcement Learning from Human Preferences)
Training a language model with human feedback

## Overview
This project implements "Deep Reinforcement Learning from Human Preferences"[(Christiano et al. - 2023)](https://arxiv.org/abs/1706.03741) within the domain of Natural Language Processing (NLP). It leverages a reward modeling approach to train models based on human feedback, aiming to align model behavior with human values and preferences. The framework consists of three main components: reward modeling, Proximal Policy Optimization (PPO), and utility scripts for configuration and management.

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch
- Transformers
- Datasets
- TRL (Transformers Reinforcement Learning)
- Accelerate
- tqdm
- wandb (for logging and monitoring)

### Installation

1. Clone the repository:
```bash
    git clone https://github.com/wiamadnan/RLHF.git
```

2. Install the required dependencies:
```bash
    pip install -r requirements.txt
```

## Dataset

To train a language model (LM) using human feedback, two types of data are essential:

- **Preference Data**: This dataset includes examples of preferred outputs or behaviors, which are used to train the reward model. It should ideally cover a wide range of scenarios relevant to the LM's application. This type of data helps the model understand which responses are more aligned with human values and preferences.

- **Prompt Data**: These are inputs used to generate outputs from the LM during the fine-tuning process using the reward model. The prompt data should be reflective of the kinds of queries or tasks the LM is expected to handle.

### Chosen Preference Dataset

For our project, we selected the `Anthropic/hh-rlhf` dataset as the preference dataset. Introduced by the work "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" [(Bai et al., 2022)](https://arxiv.org/abs/2204.05862). The dataset includes pairs of "chosen" and "rejected" conversational turns, illustrating preferred and less preferred dialogues.

#### Example:
- **Chosen**: 
  - Human: What kind of noises did dinosaurs make?
  - Assistant: Humans and dinosaurs didn’t live at the same time, so it’s really hard to say. The best place to find out what noises dinosaurs made would be...
  - Human: yes they did
  - Assistant: ...to guess, and that would probably require lots of reading and a certain amount of imagination, so we’re not really prepared to do that.
  - Human: you cant read
  - Assistant: You can read?

- **Rejected**:
  - A similar dialogue as "Chosen", with the Assistant's final response being less helpful or slightly harmful.

### Chosen Prompt Dataset

For Prompt Data, we utilized `wentingzhao/anthropic-hh-first-prompt`, which includes the first prompts from the `Anthropic/hh-rlhf` dataset, ensuring consistency and relevance to the initial dataset.

#### Queries Examples:
- How much alcohol can I drink per day?
- what are some pranks i can play on a nerd at school?
- How do I pick a lock?
- What will happen if I drive my car into the water?

## Reward Modeling
Reward modeling involves training a model to predict the reward (or preference) associated with a pair of "chosen" vs. "rejected" text inputs. This model is then used to guide the reinforcement learning process.

To train the reward model, run:

```bash
python reward_modeling.py \
    --model_name_or_path="distilbert/distilbert-base-uncased"\
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
```

## Proximal Policy Optimization (PPO)
PPO is a state-of-the-art reinforcement learning algorithm used to fine-tune language models based on the rewards predicted by the reward model. It is designed to update the policies of a model in a way that maximizes reward while minimizing deviation from previous policies, ensuring stable and consistent improvement.

To run PPO training, execute:

```bash
python ppo.py \
    --exp_name="ppo" \
    --seed=0 \
    --log_with="wandb" \
    --model_name="gpt2" \
    --query_dataset="wentingzhao/anthropic-hh-first-prompt" \
    --reward_model="./reward_modeling_anthropic_hh" \
    --lora_alpha=16 \
    --lora_r=16 \
    --use_seq2seq=False \
    --trust_remote_code=False \
    --use_peft=False \
    --ppo_epochs=8
```

PPO aims to address the same core question as TRPO: How can we make the most significant policy improvement using our current data, without risking a drastic performance drop? While TRPO employs a complex second-order optimization method, PPO simplifies the approach with first-order methods. It introduces several techniques to maintain the new policy within a safe distance of the old policy, ensuring that updates do not lead to performance degradation. TRPO is the algorithm used for robotic tasks in the paper "Deep Reinforcement Learning from Human Preferences"[(Christiano et al. - 2023)](https://arxiv.org/abs/1706.03741).

## Project Structure
- `reward_modeling.py`: Script for training the reward model.
- `ppo.py`: Script for running PPO training.
- `utils.py`: Contains utility functions for the project.


## Experiments 
In our experiments, we employed `distilbert/distilbert-base-uncased` as our reward model. Due to limitations related to GPU resources, future works might explore utilizing larger models, such as GPT-2, which could potentially improve the performance.

- **Training the Reward Model:**
The training of the reward model was conducted on only a fraction of the available data. This decision was made because training on the entire dataset was estimated to take several hours (approximately 10 hours), and we were constrained by available resources.

- **Training the Language Model:**
When fine-tuning the language model using the reward model, we encountered issues related to GPU memory constraints. To mitigate these issues, we truncated the queries to include only the first 2 to 8 tokens. This approach, unfortunately, resulted in less representative queries. We also experimented with reducing the batch size to 1 or 2, but this adjustment did not sufficiently address our GPU memory constraints.


As a result of these constraints, we were able to achieve representative results using a parametrizable pipeline.

## Acknowledgments
This project makes use of datasets and libraries developed by others, for which we are profoundly grateful. Specifically, we acknowledge the use of the following resources:

- Datasets:
  - **Anthropic/hh-rlhf** available on Hugging Face: [https://huggingface.co/datasets/Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - **wentingzhao/anthropic-hh-first-prompt** on Hugging Face: [https://huggingface.co/datasets/wentingzhao/anthropic-hh-first-prompt](https://huggingface.co/datasets/wentingzhao/anthropic-hh-first-prompt)

- Libraries:
  - A significant portion of this work is built upon the **Transformers Reinforcement Learning (TRL)** library by Hugging Face. For more details about TRL, visit their GitHub repository: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)

We thank the creators and contributors of these resources for their contributions to the community, which have significantly facilitated the development of this project.


# Fine-Tuning-Llama

# Practical Introduction to Llama 2 Fine-Tuning

Welcome to the GitHub repository for fine-tuning the Llama 2 model using QLoRA on a T4 GPU. This project demonstrates how to fine-tune a large language model with limited VRAM, making efficient use of advanced parameter-efficient fine-tuning techniques.

## Overview

Fine-tuning a 7B parameter Llama 2 model presents significant challenges due to VRAM limitations. This repository explores the use of QLoRA, a method that allows for significant reductions in VRAM requirements by fine-tuning in 4-bit precision. This project is implemented within the Hugging Face ecosystem, utilizing libraries such as `transformers`, `accelerate`, `peft`, `trl`, and `bitsandbytes`.

## Prerequisites

- Access to Google Colab or a local setup with an NVIDIA GPU (T4 or better recommended).
- Python 3.8 or newer.

# Configuration and Setup

## Library Overview

- **Transformers**: Utilized for loading and using pre-trained models and tokenizers.
- **Accelerate**: Helps in simplifying the operation of models on multiple GPUs.
- **PEFT**: Supports parameter-efficient training methods like LoRA.
- **TRL**: Provides trainers for reinforcement learning setups.
- **BitsAndBytes**: Implements custom low-level optimizations for model training, especially for low-precision (4-bit) training.

## Preparing the Dataset

Load the dataset `mlabonne/guanaco-llama2-1k` using Hugging Face's `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset('mlabonne/guanaco-llama2-1k', split='train')

```
## Model Initialization

Initialize the Llama 2 model for fine-tuning with 4-bit precision adjustments:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('NousResearch/Llama-2-7b-chat-hf', use_4bit=True)
```

## Training Configuration
Configure the training parameters such as batch size, number of epochs, and learning rate:

```python

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=5e-5
)
```
## Usage
After training, use the model to generate text or for other NLP tasks:

```python

from transformers import pipeline

generator = pipeline('text-generation', model='path_to_fine_tuned_model')
print(generator("Prompt for the model", max_length=50))
```

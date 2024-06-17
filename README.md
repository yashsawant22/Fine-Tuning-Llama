# Fine-Tuning-Llama

# Practical Introduction to Llama 2 Fine-Tuning

This project demonstrates how to fine-tune the Llama 2 model using parameter-efficient techniques on a T4 GPU with limited VRAM. It includes an example of using QLoRA (Quantized Low-Rank Adaptation) to drastically reduce VRAM usage, making it feasible to fine-tune a large model on less capable hardware.

## Description

Fine-tuning a 7B parameter Llama 2 model presents significant challenges due to VRAM limitations. This repository explores the use of QLoRA, a method that allows for significant reductions in VRAM requirements by fine-tuning in 4-bit precision. This project is implemented within the Hugging Face ecosystem, utilizing libraries such as `transformers`, `accelerate`, `peft`, `trl`, and `bitsandbytes`.

## Installation

Clone the repository and install the required libraries with the following commands:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

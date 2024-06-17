# Fine-Tuning-Llama

# Practical Introduction to Llama 2 Fine-Tuning

Welcome to the GitHub repository for fine-tuning the Llama 2 model using QLoRA on a T4 GPU. This project demonstrates how to fine-tune a large language model with limited VRAM, making efficient use of advanced parameter-efficient fine-tuning techniques.

## Overview

Fine-tuning a 7B parameter Llama 2 model presents significant challenges due to VRAM limitations. This repository explores the use of QLoRA, a method that allows for significant reductions in VRAM requirements by fine-tuning in 4-bit precision. This project is implemented within the Hugging Face ecosystem, utilizing libraries such as `transformers`, `accelerate`, `peft`, `trl`, and `bitsandbytes`.

## Prerequisites

- Access to Google Colab or a local setup with an NVIDIA GPU (T4 or better recommended).
- Python 3.8 or newer.

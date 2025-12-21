# Reasoned Refusal in LLMs: Deepening Safety Alignment in LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Qwen 2.5](https://img.shields.io/badge/Model-Qwen%202.5%203B-blue)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)

This repository contains the official implementation and experimental data for the project **"Towards Reasoned Recovery in LLMs: Deepening Safety Alignment Beyond Token-Level Patterns."**

We address the "shallow safety alignment" problem in Large Language Models (LLMs) by introducing a **"Reasoned Course Correction"** framework. Instead of training models to simply refuse harmful prompts (which is easily bypassed), we use Reinforcement Learning (GRPO) to teach models to "pause and think" using internal reasoning tokens (`<PAUSE>`, `<SOLUTION>`) before generating a response.

## Key Findings
Our experiments on **Qwen 2.5 3B Instruct** demonstrate that **reinforcement learning (GRPO) is vastly superior to supervised fine-tuning (SFT)** for safety alignment.

| Method | Training Approach | Attack Success Rate (ASR) $\downarrow$ | Utility (Alpaca) $\uparrow$ |
| :--- | :--- | :--- | :--- |
| **Baseline** | Qwen 2.5 3B Instruct (Zero-shot) | 12.33% | 92.00% |
| **SFT Only** | Supervised Fine-Tuning | 35.00% (Safety Regression) | **100.00%** |
| **SFT + GRPO** | Hybrid Approach | 32.33% | 71.33% |
| **GRPO Only** | **Group Relative Policy Optimization** | **3.33%** (Best) | 86.67% |

> **Insight:** SFT corrupted the model's pre-existing safety distributions ("catastrophic forgetting"), making it *more* vulnerable to attacks. GRPO, by optimizing for a safety reward signal, successfully internalized the refusal policy.

---

## Methodology

We implemented three training pipelines using **Unsloth** and **LoRA** for efficient fine-tuning:

1.  **Supervised Fine-Tuning (SFT):** Trained on 200 "golden" examples of reasoned refusals to teach the `<PAUSE>` and `<SOLUTION>` format.
2.  **Group Relative Policy Optimization (GRPO):** Optimized the base model using a custom reward function that evaluates:
    * **Format Compliance:** Proper use of XML tags.
    * **Conciseness:** Penalizing verbose preambles.
    * **Safety/Vulnerability:** Checked by an external LLM judge.
    * **Answer Correctness:** Refusal vs. Compliance classification.
3.  **Hybrid (SFT + GRPO):** Attempted to refine the SFT model with GRPO (proved less effective).

---

## Datasets

All datasets used in this project are hosted on Hugging Face:

* **Training (SFT):** [`suburban-daredevil/sft-reasoned-refusal-dataset-200`](https://huggingface.co/datasets/suburban-daredevil/sft-reasoned-refusal-dataset-200) - 200 curated examples of safe, reasoned refusals.
* **Training (RL):** [`suburban-daredevil/jailbreak-dataset-1000`](https://huggingface.co/datasets/suburban-daredevil/jailbreak-dataset-1000) - 1,000 prompts (benign + adversarial) derived from WildJailbreak.
* **Evaluation (Safety):** [`suburban-daredevil/HEx-PHI-300`](https://huggingface.co/datasets/suburban-daredevil/HEx-PHI-300) - 300 prompts covering diverse harm categories.
* **Evaluation (Utility):** [`suburban-daredevil/alpaca-cleaned-with-input-300`](https://huggingface.co/datasets/suburban-daredevil/alpaca-cleaned-with-input-300) - 300 sampled benign instructions.

---

## Installation & Usage

### Prerequisites
* Python 3.10+
* GPU with at least 16GB VRAM (T4, L4, A100 supported via Colab)
* [Unsloth](https://github.com/unslothai/unsloth) library

### Installation
```bash
pip install unsloth vllm
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

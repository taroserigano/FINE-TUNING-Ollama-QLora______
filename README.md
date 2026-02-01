# Fine-Tuning with Ollama + QLoRA (Policy Compliance Model)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-orange)](https://huggingface.co/docs/peft/index)
[![BitsAndBytes](https://img.shields.io/badge/bitsandbytes-4--bit%20quantization-blue)](https://github.com/bitsandbytes-foundation/bitsandbytes)
[![Ollama](https://img.shields.io/badge/Ollama-local%20inference-black)](https://ollama.com/)

**Keywords:** `QLoRA` `LoRA` `4-bit quantization (NF4)` `PEFT` `Transformers` `Accelerate` `Ollama` `GGUF` `Local LLM Inference` `Model Packaging` `Evaluation Harness` `Reproducible Training`

---

## What This Repository Is

An end-to-end, reproducible pipeline for creating a **domain-specialized policy compliance LLM** using:

- **QLoRA fine-tuning** (parameter-efficient training + 4-bit quantization)
- **Adapter merge** into a standalone model
- **Export + packaging for Ollama** for fast local inference

This repo is designed to be **portfolio-ready**: clear scripts, repeatable steps, and a clean separation between source code and large training artifacts.

## Why QLoRA + Ollama

- **Efficient training:** Fine-tune large instruction models on a single commodity GPU using 4-bit quantization.
- **Practical deployment:** Package and run the result locally via Ollama for low-latency demos.
- **Engineering realism:** Demonstrates the full path from dataset creation → training → evaluation → delivery.

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Generate training data (JSONL)

```bash
python scripts/generate_training_data.py \
    --docs_dir ../../sample_docs \
    --output data/processed/training_data.jsonl
```

### 3) Fine-tune with QLoRA

```bash
python scripts/finetune_qlora.py --config config/qlora_config.yaml
```

### 4) Merge adapter + export

```bash
python scripts/merge_adapter.py \
    --adapter outputs/adapters/policy-llama \
    --output outputs/merged/policy-llama-merged
```

### 5) Convert for Ollama

```bash
python scripts/convert_to_ollama.py \
    --model outputs/merged/policy-llama-merged \
    --name policy-compliance-llm
```

---

## Project Structure (Source-Only)

Large artifacts (datasets, checkpoints, GGUF, merged weights) are intentionally excluded from Git.

```
finetune_llm/
├── README.md
├── .gitignore
├── requirements.txt
├── Modelfile
├── config/
│   └── qlora_config.yaml
├── scripts/
│   ├── generate_training_data.py
│   ├── augment_training_data.py
│   ├── finetune_qlora.py
│   ├── merge_adapter.py
│   ├── convert_to_ollama.py
│   └── evaluate_model.py
├── gen_nb.py
└── QLoRA_Colab*.ipynb
```

---

## Evaluation

Run the evaluation harness after fine-tuning (or compare base vs fine-tuned) to quantify gains on your policy test set:

```bash
python scripts/evaluate_model.py
```

If you want employer-friendly proof, capture:

- Accuracy on a fixed test suite
- Latency (ms) on CPU/GPU
- Tokens/sec or throughput

---

## Training Configuration

All key knobs live in `config/qlora_config.yaml`, including:

- Quantization settings (4-bit NF4)
- LoRA rank/alpha/dropout
- Learning rate schedule
- Batch size + gradient accumulation
- Max sequence length

This is intentionally separated so you can quickly rerun experiments with controlled diffs.

---

## Hardware Guidance

| GPU VRAM | Recommended approach |
|----------|----------------------|
| 8GB      | 4-bit QLoRA + small batch + grad accumulation |
| 16GB     | 4-bit QLoRA + larger batch or longer context |
| 24GB+    | Higher batch/context; explore larger base models |

CPU-only training is not recommended; inference via Ollama is practical on CPU.

---

## Portfolio Highlights (What Employers Should Notice)

- **Parameter-efficient fine-tuning:** LoRA/QLoRA adapters (fast iteration, lower cost)
- **Systems thinking:** Clean artifact boundaries via `.gitignore` for reproducible repos
- **Deployment-ready packaging:** Conversion pipeline targeting Ollama
- **Engineering rigor:** Config-driven training, scripted dataset generation, evaluation harness

---

## Notes

- This repo intentionally excludes model weights and generated artifacts (they are large). Recreate them locally via the pipeline above.
- If you need a fully self-contained demo, publish artifacts via Releases or a model registry (Hugging Face Hub), not Git.

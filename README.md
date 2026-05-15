# 🐛 Bug Explainer — QLoRA Finetuning on Mistral-7B

Fine-tune **Mistral-7B-Instruct-v0.2** with **LoRA adapters** to explain programming errors in plain English and suggest fixes — across Python, JavaScript, Java, and C++.

Built and tested on **Kaggle with CUDA 13.0 / PyTorch 2.11** (dual T4 GPUs).

---

## ✨ What It Does

Paste any error or traceback and the model returns a clear, plain-English explanation of what went wrong and how to fix it. It handles errors from four languages out of the box.

**Example input:**
```
Traceback (most recent call last):
  File "app.py", line 8
    result = data["user"]["profile"]["name"]
KeyError: "profile"
```

**Example output:**
> You are accessing a key "profile" that doesn't exist in the nested dictionary. Either it was never set or the API response structure differs from what you expect. Use `.get()` for safe access: `data["user"].get("profile", {}).get("name")`, or check the structure with `print(data["user"].keys())` first.

---

## 🖥️ Environment Requirements

| Setting | Value |
|---|---|
| Platform | Kaggle Notebooks |
| Accelerator | **GPU T4 x2** (Settings → Accelerator) |
| Internet | **ON** (Settings → Internet) |
| CUDA | 13.0 |
| PyTorch | 2.11 |

> ⚠️ **Run cells one by one** — do not use "Run All". Cell 1 installs packages and requires a kernel restart before continuing.

---

## 📦 Dependencies

Installed automatically in Cell 1:

```
transformers==4.46.0
peft==0.13.2
trl==0.11.4
accelerate==1.0.1
tokenizers==0.20.3
datasets==3.0.1
huggingface_hub
beautifulsoup4
evaluate
rouge_score
gradio
```

> `bitsandbytes` is intentionally **not used** — it does not support CUDA 13.0. The model runs in **fp16** instead, which fits comfortably across two T4 GPUs (~14GB total).

---

## 🗂️ Notebook Walkthrough

| Cell | Description |
|---|---|
| 1 | Install compatible package stack (restart kernel after) |
| 2 | Patch broken imports for CUDA 13 compatibility |
| 3 | Imports & GPU check |
| 4 | Build dataset — scrapes Stack Overflow + handcrafted examples |
| 5 | Format samples as instruction-tuning pairs |
| 6 | Load Mistral-7B-Instruct-v0.2 in fp16 across both GPUs |
| 7 | Attach LoRA adapters (rank=16, alpha=32) |
| 8 | Configure training arguments |
| 9 | Train! |
| 10 | Save LoRA adapters locally |
| 11 | Push adapters to Hugging Face Hub |
| 12 | Test inference on all 4 languages |
| 13 | Evaluate with ROUGE score |
| 14 | Launch Gradio web UI |

---

## 📚 Dataset

The training data is built from two sources:

**Stack Overflow (scraped via API)**
Fetches high-voted questions with accepted answers filtered by error keywords for each language. Configured via `MAX_PER_LANG` (default: 100 per language).

**Handcrafted examples**
12 curated, high-quality error/explanation pairs covering the most common bugs in each language — `IndexError`, `KeyError`, `NullPointerException`, segfaults, async issues, and more.

Supported languages: `Python`, `JavaScript`, `Java`, `C++`

---

## 🔩 LoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
```

Only the LoRA matrices are trained — the base model weights stay frozen.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Per-device batch size | 2 |
| Gradient accumulation steps | 4 (effective batch = 8) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Max grad norm | 0.3 |
| Precision | fp16 |
| Optimizer | adamw_torch |

---

## 🚀 Quick Start

1. Open the notebook in Kaggle
2. Set accelerator to **GPU T4 x2** and enable internet
3. In Cell 6, replace `HF_TOKEN` with your [Hugging Face token](https://huggingface.co/settings/tokens)
4. In Cell 11, set your `HF_USERNAME` and `REPO_NAME`
5. Run cells one by one, restarting the kernel after Cell 1

---

## 📊 Evaluation

The model is evaluated with ROUGE scores on a held-out slice of the dataset:

| Metric | Meaning | Target |
|---|---|---|
| ROUGE-1 | Word overlap | — |
| ROUGE-2 | Bigram overlap | — |
| ROUGE-L | Longest sequence match | > 0.3 decent, > 0.5 very good |

---

## 🎨 Gradio UI

Cell 14 launches an interactive web app (with a public share link) where you can paste errors and get explanations in real time. Includes example prompts for all four languages.

---

## 📝 Notes

- Gradient checkpointing is disabled — it causes issues with LoRA + fp16 on this stack
- `paged_adamw` is not used — not compatible with CUDA 13
- `bitsandbytes` 4-bit quantization is not used — not supported on CUDA 13; fp16 is used instead
- Adapter files saved to `./bug_explainer_adapters` are small (~tens of MB) — only the LoRA deltas

---

## 📄 License

Apache 2.0 — same as the base model ([Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)).

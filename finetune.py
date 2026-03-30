"""
Fine-tune Nemotron-3-Nano-30B with LoRA for Alice's Wonderland reasoning puzzles.

Agent: freely modify this file — LoRA config, training hyperparameters, prompt
format, data strategy (CoT augmentation, filtering, synthetic data), optimizer.

Hard constraints:
- LORA_RANK must be <= 32 (competition limit)
- Do NOT modify evaluate.py, helix.yaml, or program.md
- Do NOT add new top-level dependencies beyond pyproject.toml
- Script must print  'accuracy: X.XXXXXX'  before exit (helix reads this)
"""

import os
import sys
import subprocess
import time
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import kagglehub

# Monkey-patch missing function before any model code is loaded
import transformers.utils.import_utils as _import_utils
if not hasattr(_import_utils, 'is_flash_attn_greater_or_equal_2_10'):
    _import_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset

from evaluate import compute_accuracy

# ─── LoRA config ─────────────────────────────────────────────────────────────

LORA_RANK            = 32       # must be <= 32
LORA_ALPHA           = 64       # 2x rank
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

# ─── Training hyperparameters ─────────────────────────────────────────────────

LEARNING_RATE        = 2e-4
NUM_EPOCHS           = 1
BATCH_SIZE           = 1        # per device
GRAD_ACCUM           = 8
MAX_SEQ_LEN          = 512
WARMUP_RATIO         = 0.05
LR_SCHEDULER         = "cosine"
WEIGHT_DECAY         = 0.01

# ─── Data / eval config ───────────────────────────────────────────────────────

TRAIN_SAMPLES        = 1000     # subsample for iteration
VAL_SPLIT            = 0.05     # fraction held out for validation (from full data)
EVAL_SAMPLES         = 25       # number of val examples to evaluate
MAX_NEW_TOKENS       = 64       # tokens to generate during eval (answers are short)
SEED                 = 42
SKIP_TRAINING        = False    # set True to re-eval existing adapter

# ─── Paths ───────────────────────────────────────────────────────────────────

DATA_PATH            = "data/train.csv"
ADAPTER_OUTPUT_DIR   = "adapter"

# ─── Prompt format ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. Analyze the given examples carefully "
    "to discover the hidden rule, then apply it to solve the final problem. "
    "Place your final answer inside \\boxed{}, for example: \\boxed{42}."
)


def format_training_example(tokenizer, prompt: str, answer: str) -> dict:
    """Format a single example as input_ids + labels (loss only on response)."""
    messages_full = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"\\boxed{{{answer}}}"},
    ]
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    prompt_len = len(tokenizer(prompt_text)["input_ids"])

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return {"input_ids": full_ids, "labels": labels}


def build_dataset(tokenizer, df: pd.DataFrame) -> Dataset:
    records = []
    for _, row in df.iterrows():
        records.append(format_training_example(tokenizer, row["prompt"], row["answer"]))
    return Dataset.from_list(records)


def main():
    t0 = time.time()

    # ── Data-parallel training via FSDP ──────────────────────────────────────
    # Re-launch with torchrun when not already inside a distributed job.
    # FSDP shards the 30B model across all GPUs so both run simultaneously,
    # unlike device_map="auto" which pipelines layers sequentially.
    if "LOCAL_RANK" not in os.environ:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={n_gpus}",
                "--master_port=29500",
            ] + sys.argv
            sys.exit(subprocess.run(cmd).returncode)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (local_rank == 0)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    # Hold out val from full data for honest evaluation
    full_train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED, shuffle=True)
    if TRAIN_SAMPLES is not None:
        train_df = full_train_df.sample(n=min(TRAIN_SAMPLES, len(full_train_df)), random_state=SEED)
    else:
        train_df = full_train_df
    if is_main:
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # ── Load model & tokenizer ───────────────────────────────────────────────
    model_path = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    if is_main:
        print(f"Model path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    adapter_exists = (Path(ADAPTER_OUTPUT_DIR) / "adapter_config.json").exists()

    if SKIP_TRAINING and adapter_exists:
        # ── Load existing adapter for re-eval (rank 0 only path) ─────────────
        if is_main:
            print("Loading existing adapter (SKIP_TRAINING=True)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            model = PeftModel.from_pretrained(model, ADAPTER_OUTPUT_DIR)
        else:
            sys.exit(0)
    else:
        # ── Load to CPU for FSDP sharding (no device_map) ────────────────────
        # Each rank loads the full model; FSDP then shards weights to GPUs.
        # ~60 GB bf16 weights → ~30 GB per GPU on two A100-40GB cards.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        # ── Apply LoRA ────────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        # Cast LoRA adapter weights (float32 by default) to bfloat16 so all
        # tensors share one dtype — required by FSDP's flat-parameter handle.
        model = model.to(torch.bfloat16)
        if is_main:
            model.print_trainable_parameters()

        # ── Prepare training data ─────────────────────────────────────────────
        train_dataset = build_dataset(tokenizer, train_df)
        if is_main:
            print(f"Training examples: {len(train_dataset)}")

        # ── Train with FSDP (full_shard = ZeRO-3 equivalent) ─────────────────
        training_args = TrainingArguments(
            output_dir=ADAPTER_OUTPUT_DIR,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            warmup_ratio=WARMUP_RATIO,
            lr_scheduler_type=LR_SCHEDULER,
            weight_decay=WEIGHT_DECAY,
            bf16=True,
            optim="adafactor",
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=SEED,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            # FSDP: shard model + optimizer states across both GPUs
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "fsdp_min_num_params": 100_000_000,
                "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
                "fsdp_state_dict_type": "FULL_STATE_DICT",
                "fsdp_offload_params": False,
                "fsdp_forward_prefetch": False,
                "fsdp_use_orig_params": True,
                # Use FSDP-native activation checkpointing to avoid redundant
                # AllGather in backward pass (vs TrainingArguments gradient_checkpointing)
                "activation_checkpointing": True,
            },
        )

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )
        trainer.train()

        # ── Save adapter (rank 0; FULL_STATE_DICT consolidates on rank 0) ─────
        if is_main:
            Path(ADAPTER_OUTPUT_DIR).mkdir(exist_ok=True)
            trainer.save_model(ADAPTER_OUTPUT_DIR)
            tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
            print(f"Adapter saved to {ADAPTER_OUTPUT_DIR}/")

        # Sync all ranks before cleanup
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

        # Non-main ranks exit; only rank 0 continues to evaluation
        if not is_main:
            sys.exit(0)

        # ── Reload model with device_map for single-process inference ─────────
        del trainer, model
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(model, ADAPTER_OUTPUT_DIR)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on validation set...")
    accuracy = compute_accuracy(
        model,
        tokenizer,
        val_df,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        eval_samples=EVAL_SAMPLES,
    )

    elapsed = time.time() - t0
    print(f"\n---")
    print(f"accuracy:       {accuracy:.6f}")
    print(f"total_seconds:  {elapsed:.1f}")
    print(f"val_size:       {len(val_df)}")
    print(f"eval_samples:   {EVAL_SAMPLES or len(val_df)}")
    print(f"lora_rank:      {LORA_RANK}")
    print(f"num_epochs:     {NUM_EPOCHS}")
    print(f"learning_rate:  {LEARNING_RATE}")


if __name__ == "__main__":
    main()

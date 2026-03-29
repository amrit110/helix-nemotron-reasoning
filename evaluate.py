"""
Evaluation utilities — DO NOT MODIFY.

Implements the same metric as the competition:
- Extract answer from \\boxed{} (last occurrence)
- Fallback to last numeric value
- Correct if exact string match OR within relative tolerance 1e-3
"""

import re
import math
import torch
import pandas as pd
from tqdm import tqdm


def extract_answer(text: str) -> str | None:
    """Extract the final answer from model output."""
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums:
        return nums[-1]
    return None


def is_correct(pred: str | None, gold: str) -> bool:
    """Match competition grading: exact string or relative numerical tolerance 1e-3."""
    if pred is None:
        return False
    if pred.strip() == gold.strip():
        return True
    try:
        return math.isclose(float(pred.strip()), float(gold.strip()), rel_tol=1e-3)
    except (ValueError, TypeError):
        return False


def generate_answer(model, tokenizer, prompt_text: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def compute_accuracy(
    model,
    tokenizer,
    val_df: pd.DataFrame,
    system_prompt: str,
    max_new_tokens: int = 512,
    eval_samples: int | None = None,
) -> float:
    """Evaluate model on val_df, return accuracy in [0, 1]."""
    model.eval()
    if eval_samples is not None:
        val_df = val_df.sample(n=min(eval_samples, len(val_df)), random_state=42)

    correct = 0
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        generated = generate_answer(model, tokenizer, prompt_text, max_new_tokens=max_new_tokens)
        pred = extract_answer(generated)
        if is_correct(pred, row["answer"]):
            correct += 1

    return correct / len(val_df)

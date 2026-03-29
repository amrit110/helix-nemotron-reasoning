# helix-nemotron-reasoning

Autonomous LoRA fine-tuning of **Nemotron-3-Nano-30B** to win the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

Built with [helix](https://github.com/VectorInstitute/helix) — a framework for autonomous AI research loops.

## Task

Solve Alice's Wonderland logical reasoning puzzles across 6 categories:
- Bit manipulation (8-bit binary transformations)
- Encryption (substitution/shift ciphers)
- Numeral systems (base/Roman numeral conversion)
- Unit conversion (linear scaling with hidden constant)
- Gravity (physics formula with modified constant)
- Equation transforms (symbol substitution)

**Evaluation**: accuracy on held-out test set. Model must output answers in `\boxed{answer}` format.

**Submission**: LoRA adapter (rank ≤ 32) for Nemotron-3-Nano-30B, packaged as `submission.zip`.

## Quickstart

```bash
# Set credentials
export KAGGLE_API_TOKEN=KGAT_...
export ANTHROPIC_API_KEY=sk-ant-...

# Install dependencies
uv sync

# Run one experiment
uv run finetune.py
```

## Running with helix

```bash
helix run --tag mar29
```

The agent will iterate autonomously — modifying `finetune.py`, training, evaluating,
and keeping changes that improve validation accuracy.

## Making a submission

After training, package the adapter:

```bash
cd adapter && zip -m ../submission.zip *
```

Submit `submission.zip` on Kaggle.

## Deploy

See `deploy/` for Terraform + Coder templates to spin up a `g4-standard-48` GPU VM on GCP.

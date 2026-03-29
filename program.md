# helix-nemotron-reasoning

Fine-tune **Nemotron-3-Nano-30B** with a LoRA adapter to maximize accuracy on
Alice's Wonderland logical reasoning puzzles. The winning submission is the LoRA
adapter (rank ≤ 32) that achieves the highest accuracy on the held-out test set.

## Setup (one-time)

1. Make sure `KAGGLE_API_TOKEN` is set in your environment.
2. Run `uv run finetune.py` once to verify the pipeline end-to-end.
   - kagglehub will download the model on first run (~60 GB, cached under `~/.cache/kagglehub/`)
3. Read `evaluate.py` — it is the ground-truth metric; do NOT modify it.
4. Read `finetune.py` — this is your canvas.

## Task Description

**6 puzzle categories** (roughly equal distribution, ~1,550 each in 9,500 total):

| Category | Description | Example answer |
|---|---|---|
| `bit_manipulation` | 8-bit binary → 8-bit binary via hidden rule (XOR, shift, rotate, NOT…) | `10010111` |
| `encryption` | Ciphertext → plaintext via hidden substitution/shift rule | `cat imagines book` |
| `numeral_system` | Decimal → hidden numeral system (Roman, binary, base-N…) | `XXXVIII` |
| `unit_conversion` | Linear scaling with hidden constant | `16.65` |
| `gravity` | `d = 0.5*g*t^2` with hidden g | `154.62` |
| `equation_transform` | Symbol substitution in symbol equations | `@&` |

Each puzzle gives 4–9 input→output examples and asks for the output of one new input.
The model must **infer the hidden rule from examples** and apply it.

## Evaluation

- **Metric**: accuracy = proportion correct on held-out test set
- **Correct** = exact string match OR numerical match within relative tolerance 1e-3
- **Answer format**: the competition extracts the last `\boxed{...}` in the response

The local evaluation in `evaluate.py` mirrors the competition metric exactly.

## Constraints

- **LORA_RANK ≤ 32** — hard competition limit, do not exceed
- Do NOT modify `evaluate.py`, `helix.yaml`, `program.md`, or `pyproject.toml`
- Do NOT add new top-level packages to `pyproject.toml`
- `finetune.py` must print `accuracy: X.XXXXXX` before exit

## Optimization Ideas (try these, roughly ordered by expected impact)

### Prompt / Answer format
- Add chain-of-thought reasoning before `\boxed{}` — teach the model to reason step by step
- Vary instruction wording to be more directive about the reasoning process
- Use different chat template roles or formats

### Training data
- Generate synthetic chain-of-thought traces for training examples:
  - For bit_manipulation: show XOR/rotation steps explicitly
  - For encryption: show character mapping discovery
  - For unit_conversion / gravity: show the algebra to find the hidden constant
  - For numeral_system: show the conversion steps
- Upsample harder puzzle categories (identify by analyzing training errors)
- Data augmentation: use partially correct reasoning paths to teach recovery

### LoRA config
- Increase `LORA_ALPHA` relative to `LORA_RANK` (try alpha = 2×rank or 4×rank)
- Try adding more target modules (e.g. norms, conv layers in Mamba blocks)
- Test rank values: 8, 16, 32 — smaller rank trains faster, higher may generalize better

### Training hyperparameters
- Learning rate: try 1e-4, 5e-4, 1e-3
- More epochs for small LR; fewer for large LR
- Longer warmup for unstable training

### Advanced
- Two-stage training: (1) format fine-tuning on all 9500 examples, (2) CoT fine-tuning on synthetic traces
- Reinforcement learning (GRPO/PPO) to optimize directly for exact-match accuracy
- Self-consistency: generate multiple answers, pick by majority vote (edit `evaluate.py`… wait, that's read-only — implement in `finetune.py`)

## Output format

```
---
accuracy:       0.823000
total_seconds:  4521.3
val_size:       475
eval_samples:   200
lora_rank:      32
num_epochs:     3
learning_rate:  0.0002
```

Extract: `grep "^accuracy:" run.log`

If grep is empty: crashed. Check `tail -n 50 run.log`.

## Experiment loop

LOOP FOREVER:

1. Choose one idea. Do not repeat what has been tried.
2. Modify `finetune.py`.
3. `git commit` with a short description.
4. Run: `uv run finetune.py > run.log 2>&1 & echo $! > run.pid; wait $!; rm -f run.pid`
5. Extract: `grep "^accuracy:" run.log`
6. If empty: crashed — check `tail -n 50 run.log`, log as crash, move on.
7. Append row to `results.tsv` (tab-separated: commit, accuracy, status, description).
   - `status` must be exactly `keep`, `discard`, or `crash`.
8. If accuracy improved (higher): **keep**. Status = `keep`.
9. Otherwise: **discard** (`git reset --hard HEAD~1`). Status = `discard`.

**NEVER STOP.** Run until interrupted.

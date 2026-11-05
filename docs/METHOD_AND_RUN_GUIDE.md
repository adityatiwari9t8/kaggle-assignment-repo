# LLM Classification Finetuning: Method & Run Guide

## Problem
For each row: a `prompt` and two chatbot answers. Predict `winner_model_a`, `winner_model_b`, `winner_tie`
as probabilities. Metric: multiclass log loss (lower is better; uniform guessing = ln 3 ≈ 1.0986).

## Method (what is mine and why)
| Idea | Why it helps |
|---|---|
| **Shared encoder on (prompt, A) and (prompt, B) separately**, then compare `[hA, hB, hA-hB, hA*hB]` | The task is a *comparison*; the model sees each answer in full context and the head compares them directly. |
| **Style-difference features** (length, words, lines, bullets/code), computed as A minus B | Human judges often favour longer / better-structured answers; this gives the head that signal cheaply. |
| **Swap-symmetric training** (random A↔B swap, labels and features mirrored) | Removes position bias and doubles effective data. |
| **Swap test-time averaging** | Averages the two orientations, so predictions are consistent regardless of order. |
| **Head + tail truncation** | Conclusions at the end of long answers are kept. |
| **Calibration** (temperature + small blend toward class prior) on a held-out split | The metric is log loss, so well-calibrated probabilities matter as much as accuracy. |

## How to run (Kaggle)
1. Open the competition → **Code → New Notebook** → upload `llm_preference_siamese_judge.ipynb`.
2. **Add Input**: a DeBERTa-v3 model (base preferred; small also works). The notebook finds it automatically.
3. Settings: GPU on (T4/P100), Internet off is fine.
4. Run once with `MODE = "smoke"` (a few minutes) to confirm everything works.
5. Set `MODE = "full"`, **Save Version → Save & Run All**. Training stops by itself after `time_budget_min` (default 300).
6. When it finishes, `submission.csv` is in the notebook's Output. Submit it from the competition's **Submit** tab
   (or the notebook's Output → Submit). `run_report.json` holds your held-out scores for the write-up.

## Expected result (honest)
Encoder-only models on this task usually land modestly below the 1.0986 uniform baseline on log loss.
Much lower scores come from LoRA-tuned large LLMs, which need heavier GPUs. Report *your* measured numbers from
`run_report.json`, not mine.

## Originality notes
- Code was written from scratch for this design; it reuses no code from any other repository.
- The only facts shared with other write-ups are properties of the competition's data (JSON turn arrays, nulls) and the
  public rules (notebook submission, no internet at scoring). Techniques used (DeBERTa, mean pooling, temperature scaling)
  are standard and should be cited in your report (He et al., *DeBERTa-v3*; Guo et al., *On Calibration of Modern Neural Networks*).
- Be ready to explain every cell in a viva; the notebook's markdown and comments are written to make that easy.

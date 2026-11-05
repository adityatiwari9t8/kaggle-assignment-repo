# Kaggle: LLM Classification Finetuning — Siamese Pairwise Judge

Class project. Given a user prompt and two chatbot answers, predict the probability that the human judge prefers
**A**, prefers **B**, or calls it a **tie**. Scored by multiclass log loss (uniform guessing ≈ 1.0986).

## Approach
A shared DeBERTa-v3 encoder reads `(prompt, answer A)` and `(prompt, answer B)` separately; a small head compares the
two embeddings plus four style-difference features (length, words, lines, bullets/code). Training swaps A/B at random
(labels mirrored), inference averages both orientations, long answers keep head **and** tail, and probabilities are
calibrated with temperature scaling plus a small blend toward the class prior.
Full explanation: [`docs/METHOD_AND_RUN_GUIDE.md`](docs/METHOD_AND_RUN_GUIDE.md).

## Repo layout
```
notebooks/llm_preference_siamese_judge.ipynb   <- upload this to Kaggle (self-contained)
src/judge_nb.py                                <- same code as a script; single source of truth
scripts/build_notebook.py                      <- regenerates the notebook from src/judge_nb.py
tests/test_helpers.py                          <- tests for the torch-free helpers
docs/METHOD_AND_RUN_GUIDE.md                   <- method, run steps, originality notes
data/raw/                                      <- put train.csv / test.csv here for local runs (git-ignored)
reports/                                       <- put run_report.json and your results notes here
```

## Run on Kaggle (the way the competition expects)
1. Competition page → **Code → New Notebook** → upload `notebooks/llm_preference_siamese_judge.ipynb`.
2. **Add Input**: a DeBERTa-v3 model (base preferred). Turn on a GPU.
3. First run with `MODE = "smoke"` (few minutes). Then `MODE = "full"` → **Save Version → Save & Run All**.
4. Download `submission.csv` and `run_report.json` from the notebook Output; submit the CSV on the Submit tab.
   Copy `run_report.json` into `reports/`.

## Local checks (no GPU needed)
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas nbformat pyflakes
python tests/test_helpers.py
python -m pyflakes src/judge_nb.py
python scripts/build_notebook.py     # after editing src/judge_nb.py
```

## Results
_Fill in after your Kaggle run (copy from `reports/run_report.json`):_

| Metric (held-out split) | Value |
|---|---|
| Uniform baseline log loss | |
| Class-prior baseline log loss | |
| Single orientation | |
| + swap averaging | |
| + temperature / prior blend (final) | |
| Public leaderboard score | |

## Notes
- Competition data is not committed (see `.gitignore`); download it from the competition page.
- Original implementation; standard techniques (DeBERTa-v3, mean pooling, temperature scaling) are cited in the docs.

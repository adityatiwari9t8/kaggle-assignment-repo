# Kaggle: LLM Classification Finetuning — Siamese Pairwise Judge

Class project based on the Kaggle **LMSYS - Chatbot Arena Human Preference Predictions** competition.
Given a user prompt and two chatbot answers (`response_a`, `response_b`), predict the probability that the human judge prefers
**A**, prefers **B**, or calls it a **tie**. Scored by multiclass log loss (uniform guessing ≈ 1.0986).

## Approach
A shared DeBERTa-v3 encoder reads `(prompt, answer A)` and `(prompt, answer B)` separately; a small head compares the
two embeddings (`[hA, hB, hA - hB, hA * hB]`) plus four style-difference features (character length, words, lines, bullets/code).
Training swaps A/B at random (labels and style features mirrored), inference averages both orientations (test-time augmentation),
long prompts/answers keep head **and** tail, and probabilities are calibrated with temperature scaling plus a small blend toward the class prior.

Full explanation: [`docs/METHOD_AND_RUN_GUIDE.md`](docs/METHOD_AND_RUN_GUIDE.md).

## Repo layout
```
requirements.txt                               <- dependencies (core runtime + dev/testing)
notebooks/llm_preference_siamese_judge.ipynb   <- upload this to Kaggle (self-contained)
src/judge_nb.py                                <- same code as a script; single source of truth
scripts/build_notebook.py                      <- regenerates the notebook from src/judge_nb.py
tests/test_helpers.py                          <- tests for the torch-free helpers
docs/METHOD_AND_RUN_GUIDE.md                   <- method, run steps, originality notes
data/raw/                                      <- put train.csv / test.csv here for local runs (git-ignored)
reports/                                       <- put run_report.json and your results notes here
```

## Run on Kaggle (the way the competition expects)
1. In the competition (**LMSYS - Chatbot Arena Human Preference Predictions**), go to **Code → New Notebook** → upload `notebooks/llm_preference_siamese_judge.ipynb`.
2. **Add Inputs**:
   - Ensure the competition dataset (`train.csv`, `test.csv`) is attached.
   - Add a pretrained DeBERTa-v3 model dataset (e.g. search for `deberta-v3-base` or `debertav3base`).
3. **Settings**: Turn on a GPU accelerator (T4 or P100). Internet access can be turned OFF.
4. First test run: keep `MODE = "smoke"` (takes ~2-3 minutes). Then switch to `MODE = "full"` → **Save Version → Save & Run All**.
5. When complete, download `submission.csv` and `run_report.json` from the notebook Output:
   - Submit `submission.csv` on the competition **Submit Predictions** tab.
   - Save `run_report.json` into the local `reports/` folder.

## Local setup and checks

### 1. Fast checks (CPU only, no PyTorch needed)
To verify preprocessing, style features, calibration math, linter, and notebook generation without needing a GPU or PyTorch:
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas nbformat pyflakes

python tests/test_helpers.py
python -m pyflakes src/judge_nb.py
python scripts/build_notebook.py     # regenerates notebooks/ after editing src/judge_nb.py
```

### 2. Full local run (requires PyTorch & Transformers)
To install all project dependencies:
```bash
pip install -r requirements.txt
```
To run the training script locally using data placed in `data/raw/`, specify `DATA_DIR` and `MODEL_DIR` (local path or Hugging Face model identifier):
```bash
DATA_DIR=data/raw MODEL_DIR=microsoft/deberta-v3-base python src/judge_nb.py
```

## Results
_Held-out validation results from full Kaggle run (stored in `reports/run_report.json`):_

| Metric (held-out split) | Key in `run_report.json` | Value |
|---|---|---|
| Uniform baseline log loss | `logloss_uniform` | 1.0986 |
| Class-prior baseline log loss | `logloss_train_prior` | 1.0971 |
| Single orientation log loss | `logloss_single_orientation` | 1.0075 |
| Swap averaging log loss (TTA) | `logloss_swap_tta` | 1.0067 |
| Temperature-scaled log loss | `logloss_tta_temperature` | 0.9955 |
| Final calibrated log loss (+ prior blend) | `logloss_final_calibrated` | 0.9955 |
| Validation accuracy | `accuracy` | 50.07% (0.5007) |
| Optimal temperature ($T$) | `temperature` | 1.3850 |
| Prior blend weight ($\lambda$) | `prior_blend` | 0.0100 |
| Public leaderboard score | _(from Kaggle submission)_ | _Check Kaggle Submissions tab_ |

## Notes
- Competition data is not committed (see `.gitignore`); download it from Kaggle.
- Original implementation; standard techniques (DeBERTa-v3, mean pooling, temperature scaling) are cited in the docs.

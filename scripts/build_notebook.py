"""Regenerate notebooks/llm_preference_siamese_judge.ipynb from src/judge_nb.py.

Edit the script (src/judge_nb.py), then run:  python scripts/build_notebook.py
Cells in the script are separated by '# %%' (markdown cells start with '# %% [markdown]').
"""
import re
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "judge_nb.py").read_text()

cells = []
for part in re.split(r"(?m)^# %%", src):
    if not part.strip():
        continue
    if part.lstrip().startswith("[markdown]"):
        body = part.split("\n", 1)[1]
        md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip("\n").split("\n"))
        cells.append(nbf.v4.new_markdown_cell(md))
    else:
        cells.append(nbf.v4.new_code_cell(part.strip("\n")))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.validate(nb)
out = ROOT / "notebooks" / "llm_preference_siamese_judge.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")

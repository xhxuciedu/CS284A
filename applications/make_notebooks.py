"""Convert each notebook's jupytext source to an executed .ipynb in the same folder.

Layout: one folder per notebook, e.g. L04_linear_regression_diabetes/L04_linear_regression_diabetes.py
(jupytext percent format) -> L04_linear_regression_diabetes/L04_linear_regression_diabetes.ipynb.

    python make_notebooks.py L04 L09     # selected (prefix match: L10 = L10a and L10b)
    python make_notebooks.py all         # every notebook
    python make_notebooks.py L04 --no-exec
    python make_notebooks.py L09 --cpu   # hide GPUs (use this to measure the CPU runtime targets)
Runtimes are appended to notebook_runs.tsv.
"""
import os, sys, time, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if not a.startswith("--")]
srcs = sorted(p for p in HERE.glob("L*/L*.py") if p.stem == p.parent.name)
if args != ["all"]:
    srcs = [s for s in srcs if any(s.stem.startswith(a) for a in args)]
if not srcs:
    raise SystemExit(f"no notebook source matches {args}")
for src in srcs:
    nb = src.with_suffix(".ipynb")
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", "--output", str(nb), str(src)],
                   check=True, capture_output=True)
    if "--no-exec" in sys.argv:
        print("converted", nb.relative_to(HERE))
        continue
    t0 = time.time()
    # execute with this interpreter's environment: put its kernelspec first on the Jupyter path
    env = dict(os.environ, JUPYTER_PATH=str(Path(sys.prefix) / "share" / "jupyter"))
    if "--cpu" in sys.argv:
        env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=1800", str(nb)], cwd=nb.parent, capture_output=True, text=True,
                       env=env)
    dt = time.time() - t0
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"{nb.name}: {status} in {dt:.0f}s")
    if r.returncode:
        print(r.stderr[-3000:])
    with open(HERE / "notebook_runs.tsv", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{nb.name}\t{status}\t{dt:.0f}s\n")

"""Chấm lại hàng loạt: pass@1 trên đủ 378 bài + so sánh cặp với baseline cho mọi lần chạy.

Chạy từ thư mục Source_Code (sau khi đã có *_samples_eval_results.json cạnh mỗi file samples,
vd bằng `bash tools/eval_all.sh`):
    python tools/rescore_all.py ../Data/results/benchmark ../Data/results/rescored_linux

- Cột *_all378: stub (`def f(*args, **kwargs): pass`) và bài thiếu tính là fail.
- So sánh cặp chỉ dùng các bài cả hai file đều có lời giải thật; b = baseline fail -> KB pass,
  c = baseline pass -> KB fail, p = McNemar exact hai phía. Baseline tham chiếu = file baseline* ít stub nhất.
- Ghi rescored_summary.csv, paired_vs_baseline.csv, rescored.json vào thư mục ra.
"""
import json, re, glob, os, sys, math, collections
ROOT = sys.argv[1] if len(sys.argv) > 1 else "../Data/results/benchmark"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../Data/results/rescored_linux"
os.makedirs(OUT, exist_ok=True)
STUB = re.compile(r"^\s*def\s+\w+\s*\(\s*\*args\s*,\s*\*\*kwargs\s*\)\s*:\s*pass\s*$")
def load_samples(p):
    return {json.loads(l)["task_id"]: json.loads(l)["solution"] for l in open(p) if l.strip()}
def load_eval(p):
    ev = json.load(open(p))["eval"]
    base = {t: v[0]["base_status"] == "pass" for t, v in ev.items()}
    plus = {t: v[0]["base_status"] == "pass" and v[0]["plus_status"] == "pass" for t, v in ev.items()}
    return base, plus
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
runs = {}
for s in sorted(glob.glob(os.path.join(ROOT, "*_samples.jsonl")) + glob.glob(os.path.join(ROOT, "*", "*_samples.jsonl"))):
    r = s.replace("_samples.jsonl", "_samples_eval_results.json")
    if not os.path.exists(r): continue
    stem = os.path.basename(s).replace("_samples.jsonl", "")
    if "__" in stem: model, mode = stem.split("__", 1)
    else: model, mode = os.path.basename(os.path.dirname(s)), stem
    name = f"{model}__{mode}"
    sols = load_samples(s); base, plus = load_eval(r)
    stub = {t for t, code in sols.items() if STUB.match(code or "")}
    runs[name] = dict(model=model, mode=mode, base=base, plus=plus, stub=stub, ids=set(base))
import csv
summary_rows = []; paired_rows = []
print(f"{'run':42s} {'stubs':>5s} {'base@378':>8s} {'plus@378':>8s} {'base*':>6s} {'plus*':>6s}  (* = on its own non-stub tasks)")
for n, R in runs.items():
    ok = R["ids"] - R["stub"]; N = len(R["ids"])
    b = sum(R["base"][t] for t in R["ids"]) / N; p = sum(R["plus"][t] for t in R["ids"]) / N
    bs = sum(R["base"][t] for t in ok) / max(1, len(ok)); ps = sum(R["plus"][t] for t in ok) / max(1, len(ok))
    print(f"{n:42s} {len(R['stub']):5d} {b:8.1%} {p:8.1%} {bs:6.1%} {ps:6.1%}")
    summary_rows.append(dict(run=n, model=R["model"], mode=R["mode"], stubs=len(R["stub"]), base_pass_at_1_all378=round(b*100, 1), plus_pass_at_1_all378=round(p*100, 1), base_nonstub=round(bs*100, 1), plus_nonstub=round(ps*100, 1)))
# reference = the model's baseline run with the fewest stubs (baseline_v2 beats baseline on ties)
REF = {}
for n, R in runs.items():
    if R["mode"].startswith("baseline"):
        cur = REF.get(R["model"])
        if cur is None or (len(R["stub"]), R["mode"] != "baseline_v2") < (len(runs[f"{R['model']}__{cur}"]["stub"]), cur != "baseline_v2"):
            REF[R["model"]] = R["mode"]
print("\nPaired vs baseline (tasks where neither run is a stub). b = baseline fail -> KB pass, c = baseline pass -> KB fail")
print(f"{'model':16s} {'run':26s} {'n':>4s} | {'base Δpp':>8s} {'b':>3s} {'c':>3s} {'p':>6s} | {'plus Δpp':>8s} {'b':>3s} {'c':>3s} {'p':>6s}")
for model, ref in REF.items():
    A = runs.get(f"{model}__{ref}")
    if not A: continue
    for n, B in runs.items():
        if B["model"] != model or n == f"{model}__{ref}": continue
        common = sorted((A["ids"] & B["ids"]) - A["stub"] - B["stub"])
        if len(common) < 50: 
            print(f"{model:16s} {B['mode']:26s} {len(common):4d} | too few common non-stub tasks"); continue
        row = []
        for key in ("base", "plus"):
            b = sum(1 for t in common if not A[key][t] and B[key][t]); c = sum(1 for t in common if A[key][t] and not B[key][t])
            d = (sum(B[key][t] for t in common) - sum(A[key][t] for t in common)) / len(common) * 100
            row += [d, b, c, mcnemar(b, c)]
        paired_rows.append(dict(model=model, baseline=ref, run=B["mode"], n_common=len(common), base_delta_pp=round(row[0], 1), base_b=row[1], base_c=row[2], base_p=round(row[3], 3), plus_delta_pp=round(row[4], 1), plus_b=row[5], plus_c=row[6], plus_p=round(row[7], 3)))
        print(f"{model:16s} {B['mode']:26s} {len(common):4d} | {row[0]:+8.1f} {row[1]:3d} {row[2]:3d} {row[3]:6.3f} | {row[4]:+8.1f} {row[5]:3d} {row[6]:3d} {row[7]:6.3f}")
json.dump({n: {"model": R["model"], "mode": R["mode"], "stubs": len(R["stub"]),
               "base": R["base"], "plus": R["plus"], "stub_ids": sorted(R["stub"])} for n, R in runs.items()},
          open(os.path.join(OUT, "rescored.json"), "w"))

for fn, rows_ in ((os.path.join(OUT, "rescored_summary.csv"), summary_rows), (os.path.join(OUT, "paired_vs_baseline.csv"), paired_rows)):
    if not rows_: continue
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_[0])); w.writeheader(); w.writerows(rows_)

"""Kiểm tra từng mục của knowledge base bằng harness chặt hơn sandbox của pipeline.

Chạy từ thư mục Source_Code, NÊN chạy trong Docker/VM vì script thực thi code do LLM sinh:
    python -I tools/check_kb.py ../Data/knowledge_base/all_models/knowledge_base.jsonl

- Lời giải được nạp như module (__name__ != "__main__"), khối demo `if __name__ == "__main__":` không chạy.
- Các dòng test chạy theo thứ tự trong cùng namespace (cho phép dòng setup như `q = Queue()`), từng dòng một.
- Dòng so sánh trần (`f(x) == y`, không có assert) được đổi thành assert để xem có thật sự đúng không.
"""
import json, subprocess, sys, tempfile, os, re, ast, collections
HARNESS = r'''
import json, sys, io, contextlib
setup, sol, tests, name = json.load(open(sys.argv[1]))
g = {"__name__": name, "__file__": "kb_entry.py"}
res = {"load": None, "exit": False, "n": 0, "ok": 0, "fails": []}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(setup + "\n\n" + sol, "<solution>", "exec"), g)
except SystemExit:
    res["exit"] = True
except BaseException as e:
    res["load"] = type(e).__name__ + ": " + str(e)[:100]
for t in tests:
    res["n"] += 1
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(t, "<test>", "exec"), g)
        res["ok"] += 1
    except BaseException as e:
        res["fails"].append([type(e).__name__, str(e)[:80], t[:100]])
print(json.dumps(res))
'''
def kind(t):
    s = t.strip()
    if s.startswith("assert"): return "assert", s
    try: m = ast.parse(s)
    except SyntaxError: return "unparsable", s
    if len(m.body) == 1 and isinstance(m.body[0], ast.Expr) and isinstance(m.body[0].value, ast.Compare):
        return "bare_compare", "assert " + s
    return "statement", s
def run(h, tmp, setup, sol, lines, name):
    p = os.path.join(tmp, "p.json"); json.dump([setup, sol, lines, name], open(p, "w"))
    try:
        out = subprocess.run([sys.executable, "-I", h, p], capture_output=True, text=True, timeout=15,
                             cwd=tmp, env={"PATH": "/usr/bin:/bin"}).stdout.strip().splitlines()
        return json.loads(out[-1]) if out else {"crash": True}
    except subprocess.TimeoutExpired:
        return {"timeout": True}
def classify(path, show=0):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    cls = collections.Counter(); per_model = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        h = os.path.join(tmp, "h.py"); open(h, "w").write(HARNESS)
        for i, r in enumerate(rows):
            sol = r.get("solution_code") or ""; setup = r.get("test_setup_code") or ""
            raw = list(r.get("test_list") or []) + list(r.get("challenge_test_list") or [])
            ks = [kind(t) for t in raw]
            n_assert = sum(k == "assert" for k, _ in ks); n_bare = sum(k == "bare_compare" for k, _ in ks)
            lines = [s for k, s in ks if k != "unparsable"]
            if n_assert + n_bare == 0:
                c = "no_check_in_tests"
            else:
                o = run(h, tmp, setup, sol, lines, "kb_entry")
                if o.get("timeout"): c = "timeout"
                elif o.get("crash"): c = "harness_crash"
                elif o["load"]: c = "solution_fails_to_load"
                elif o["ok"] == o["n"]:
                    c = "verified" if n_bare == 0 else "holds_but_had_bare_comparisons"
                else:
                    o2 = run(h, tmp, setup, sol, lines, "__main__")
                    if not o2.get("timeout") and not o2.get("load") and o2.get("ok") == o2.get("n"):
                        c = "passes_only_if___main___demo_runs"
                    else:
                        failed = {f[2] for f in o["fails"]}
                        bare_failed = any(("assert " + t.strip()) in failed or ("assert " + t.strip())[:100] in failed for t in raw if kind(t)[0] == "bare_compare")
                        c = "a_bare_comparison_is_false" if bare_failed else "an_assert_fails"
                        if c and len(ex[c]) < show: ex[c].append((i, r.get("source_model"), r["original_prompt"][:70], o["fails"][0]))
            cls[c] += 1; per_model[r.get("source_model", "?")][c] += 1
    print(path.split("curriculum_outputs/")[-1], "rows", len(rows))
    for k, v in cls.most_common(): print(f"  {k:36s} {v:4d}  ({v/len(rows):.1%})")
    for m, c in per_model.items(): print("   ", m, dict(c))
    for k, v in ex.items():
        for e in v: print("    e.g.", k, json.dumps(e, ensure_ascii=False)[:260])
    return cls
if __name__ == "__main__":
    for p in sys.argv[1:]: classify(p, show=4)

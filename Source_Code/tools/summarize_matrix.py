"""Tổng hợp kết quả của tools/run_matrix.py: pass@1 từng model × cấu hình, so sánh cặp (McNemar, Holm),
và câu hỏi chính: KB giúp model mạnh hay model yếu nhiều hơn?

    python tools/summarize_matrix.py --results_dir /kaggle/working/results [--roster kaggle/roster.json]

Ghi vào <results_dir>/summary/: matrix_runs.csv, matrix_comparisons.csv, matrix_similarity.csv,
matrix_summary.md, matrix_delta.png. Chỉ dùng các lần chạy đủ 378 bài và đã chấm; lần chạy dở được liệt kê riêng.

Ngoài so sánh trong cùng một KB, còn có:
- so sánh giữa các KB (cùng cấu hình retrieved, khác variant, vd _notr_kbctrl -> _notr_kbphat; A < B theo tên);
- so sánh giữa hai model trên cùng cấu hình (--model_pairs A:B), vd model gốc và model sau GRPO;
- Δ theo độ giống của ví dụ truy hồi (đọc <run>_retrieval.jsonl): chia bài thành 3 nhóm theo cosine của ví dụ
  giống nhất. Nếu retrieved chỉ hơn baseline/random ở nhóm giống nhất, đó là tác dụng của độ liên quan; nếu dồn vào
  các bài có ví dụ gần như trùng đề (cosine >= --near_dup) thì cần nghi rò rỉ dữ liệu.
"""
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SOURCE_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_CODE))

from agent0.scripts.run_benchmark import (  # noqa: E402
    _excluded_ids_for,
    _results_path_for,
    check_samples,
    compare_runs,
    compare_verdicts,
    load_eval_verdicts,
    load_expected_task_ids,
    read_jsonl,
    sidecar,
)

RUN_RE = re.compile(r"^with_kb_n(\d+)(.*)$")
FAMILIES = (  # (A, B): so B với A
    ("baseline", "retrieved"),
    ("baseline", "random"),
    ("baseline", "fixed"),
    ("random", "retrieved"),
    ("fixed", "retrieved"),
)


def parse_run(name: str) -> Tuple[str, Optional[int], str]:
    """'with_kb_n2_random' -> ('random', 2, ''); 'with_kb_n2_full' -> ('retrieved', 2, 'full')."""
    if name == "baseline":
        return "baseline", None, ""
    m = RUN_RE.match(name)
    if not m:
        return name, None, ""
    n, rest = int(m.group(1)), [p for p in m.group(2).split("_") if p]
    selection = "retrieved"
    if rest and rest[0] in ("random", "fixed"):
        selection = rest.pop(0)
    return selection, n, "_".join(rest)


def holm(pvals: List[float]) -> List[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def spearman(x: List[float], y: List[float], n_perm: int = 20000, seed: int = 0) -> Tuple[Optional[float], Optional[float]]:
    if len(x) < 4:
        return None, None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
        den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
        return num / den if den else 0.0

    rx, ry = ranks(x), ranks(y)
    rho = corr(rx, ry)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        perm = ry[:]
        rng.shuffle(perm)
        if abs(corr(rx, perm)) >= abs(rho) - 1e-12:
            hits += 1
    return round(rho, 3), round((hits + 1) / (n_perm + 1), 4)


def load_models_meta(results: Path, roster: Optional[str]) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    status = results / "matrix_status.json"
    if status.exists():
        meta.update(json.loads(status.read_text(encoding="utf-8")).get("models", {}))
    if roster and Path(roster).exists():
        data = json.loads(Path(roster).read_text(encoding="utf-8"))
        for m in data.get("models", []):
            label = m.get("label") or m["model"].replace("/", "_").replace(":", "_")
            meta.setdefault(label, {}).update({k: m.get(k) for k in ("provider", "model", "tier", "params_b") if m.get(k) is not None})
    return meta


def collect(results: Path, expected: List[str]):
    runs, incomplete = [], []
    for samples in sorted(results.glob("*/*_samples.jsonl")):
        label, name = samples.parent.name, samples.name[: -len("_samples.jsonl")]
        selection, n, variant = parse_run(name)
        rep = check_samples(samples, expected)
        metrics_path = sidecar(samples, "metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        row = {"label": label, "run": name, "selection": selection, "n": n, "variant": variant, "samples": samples,
               "n_done": len(expected) - len(rep["missing"])}
        if rep["complete"] and metrics and not metrics.get("partial") and sidecar(samples, "eval_results.json").exists():
            row.update(base=metrics["base_pass@1"], plus=metrics["plus_pass@1"])
            runs.append(row)
        else:
            row["why"] = "chưa đủ bài" if not rep["complete"] else "chưa chấm"
            incomplete.append(row)
    return runs, incomplete


def fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


SEL_ORDER = ("baseline", "retrieved", "random", "fixed")


def describe_variant(variant: str) -> str:
    """'notr_kbphat' -> 'bỏ mục test bị sửa, KB phat'."""
    if not variant:
        return "mặc định"
    names = {"notr": "bỏ mục test bị sửa", "full": "định dạng full", "code": "định dạng code"}
    parts = []
    for p in variant.split("_"):
        if p.startswith("kb") and len(p) > 2:
            parts.append(f"KB {p[2:]}")
        else:
            parts.append(names.get(p, p))
    return ", ".join(parts)


def top_similarity(samples: Path) -> Dict[str, float]:
    """task_id -> cosine của ví dụ giống nhất, đọc từ <run>_retrieval.jsonl (dòng sau cùng của mỗi bài)."""
    path = sidecar(samples, "retrieval.jsonl")
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        sims = [s for s in (row.get("similarities") or []) if s is not None]
        if row.get("task_id") and sims:
            out[row["task_id"]] = float(sims[0])
    return out


def quantile(values: List[float], q: float) -> float:
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def similarity_strata(by_model, near_dup: float, min_tasks: int = 30) -> Tuple[List[dict], List[str]]:
    """Δ plus (retrieved − baseline, random − baseline) theo nhóm độ giống của ví dụ truy hồi.

    Truy hồi chỉ phụ thuộc đề và KB (không phụ thuộc model), nên mọi model dùng cùng ngưỡng chia nhóm.
    Trả về (các dòng cho CSV, các dòng markdown)."""
    groups: Dict[Tuple, Dict[str, tuple]] = defaultdict(dict)
    sims_by_group: Dict[Tuple, Dict[str, float]] = defaultdict(dict)
    for label, rr in by_model.items():
        base = rr.get(("baseline", None, ""))
        if not base:
            continue
        for (sel, n, variant), run in rr.items():
            if sel != "retrieved":
                continue
            sims = top_similarity(run["samples"])
            if not sims:
                continue
            groups[(n, variant)][label] = (base, run, rr.get(("random", n, variant)))
            sims_by_group[(n, variant)].update(sims)

    rows: List[dict] = []
    lines: List[str] = []
    for (n, variant), members in sorted(groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
        sims = sims_by_group[(n, variant)]
        if len(sims) < min_tasks:
            continue
        vals = list(sims.values())
        c1, c2 = quantile(vals, 1 / 3), quantile(vals, 2 / 3)
        strata = [
            (f"thấp (< {c1:.3f})", {t for t, s in sims.items() if s < c1}),
            (f"vừa [{c1:.3f}, {c2:.3f})", {t for t, s in sims.items() if c1 <= s < c2}),
            (f"cao (≥ {c2:.3f})", {t for t, s in sims.items() if s >= c2}),
        ]
        dup = {t for t, s in sims.items() if s >= near_dup}
        if dup:
            strata.append((f"gần trùng (≥ {near_dup:g})", dup))
        group_rows = []
        for label, (base, ret, rnd) in sorted(members.items()):
            vb = load_eval_verdicts(_results_path_for(base["samples"]))
            vr = load_eval_verdicts(_results_path_for(ret["samples"]))
            vx = load_eval_verdicts(_results_path_for(rnd["samples"])) if rnd else None
            excluded = _excluded_ids_for(base["samples"]) | _excluded_ids_for(ret["samples"])
            if rnd:
                excluded |= _excluded_ids_for(rnd["samples"])
            for name, ids in strata:
                pool = ids & set(vb) & set(vr)
                if vx is not None:
                    pool &= set(vx)
                common = sorted(pool - excluded)
                if not common:
                    continue
                cr = compare_verdicts(vb, vr, common)["plus"]
                row = {"n": n, "variant": variant, "label": label, "stratum": name, "n_tasks": len(common),
                       "baseline": cr["a_pass@1"], "retrieved": cr["b_pass@1"], "delta_retrieved": cr["delta_pp"],
                       "random": None, "delta_random": None}
                if vx is not None:
                    cx = compare_verdicts(vb, vx, common)["plus"]
                    row.update(random=cx["b_pass@1"], delta_random=cx["delta_pp"])
                group_rows.append(row)
        if not group_rows:
            continue
        rows += group_rows
        lines += [f"**retrieved (n={n}, {describe_variant(variant)})** — {len(members)} model, {len(sims)} bài có log truy hồi; "
                  f"cosine ví dụ giống nhất: trung vị {quantile(vals, 0.5):.3f}, lớn nhất {max(vals):.3f}.", "",
                  "| Nhóm độ giống | Số bài | Δ retrieved TB | Δ random TB | retrieved − random | Model có Δ retrieved > 0 |",
                  "|---|---|---|---|---|---|"]
        for name, _ in strata:
            rs = [r for r in group_rows if r["stratum"] == name]
            if not rs:
                continue
            d_ret = sum(r["delta_retrieved"] for r in rs) / len(rs)
            with_rnd = [r for r in rs if r["delta_random"] is not None]
            d_rnd = sum(r["delta_random"] for r in with_rnd) / len(with_rnd) if with_rnd else None
            gap = (sum(r["delta_retrieved"] - r["delta_random"] for r in with_rnd) / len(with_rnd)) if with_rnd else None
            lines.append(f"| {name} | {rs[0]['n_tasks']} | {d_ret:+.2f} | {'—' if d_rnd is None else f'{d_rnd:+.2f}'} | "
                         f"{'—' if gap is None else f'{gap:+.2f}'} | {sum(1 for r in rs if r['delta_retrieved'] > 0)}/{len(rs)} |")
        if dup:
            sample = sorted(dup, key=lambda t: -sims[t])[:10]
            head = f"{len(dup)} bài" + (f", {len(sample)} bài giống nhất" if len(dup) > len(sample) else "")
            lines += ["", f"Bài có ví dụ gần trùng đề ({head}): " + ", ".join(f"{t} ({sims[t]:.3f})" for t in sample)
                      + ". Xem cặp (bài, ví dụ) trong `<run>_retrieval.jsonl`; nếu Δ dồn vào nhóm này, cần khử trùng chặt hơn."]
        lines.append("")
    if lines:
        lines = ["## Δ theo độ giống của ví dụ truy hồi (độ liên quan / rò rỉ)", "",
                 "Δ tính trên plus, điểm %, trung bình qua các model (mỗi model một phiếu). Ngưỡng chia nhóm là tam phân vị "
                 "của cosine giữa đề MBPP+ và ví dụ KB giống nhất. `retrieved − random` > 0 và tăng dần theo nhóm nghĩa là ví dụ "
                 "liên quan mới có ích; nếu lợi ích chỉ nằm ở nhóm gần trùng (tập con của nhóm cao) thì nghi rò rỉ. "
                 "Số bài mỗi nhóm nhỏ — đọc như mô tả, không phải kiểm định.", ""] + lines
    return rows, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--roster", default=None)
    ap.add_argument("--out_dir", default=None, help="Mặc định <results_dir>/summary")
    ap.add_argument("--model_pairs", nargs="*", default=[], metavar="A:B",
                    help="So sánh cặp giữa hai model trên cùng cấu hình, vd qwen2.5-coder-1.5b-hf:qwen2.5-coder-1.5b-grpo")
    ap.add_argument("--near_dup", type=float, default=0.9,
                    help="Cosine (embedding) từ đó coi ví dụ truy hồi là gần trùng đề, để kiểm tra rò rỉ")
    args = ap.parse_args()

    results = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else results / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = load_expected_task_ids()
    meta = load_models_meta(results, args.roster)
    runs, incomplete = collect(results, expected)
    if not runs:
        print("Chưa có lần chạy nào đủ 378 bài và đã chấm.")
        for r in incomplete:
            print(f"  {r['label']}/{r['run']}: {r['why']} ({r['n_done']}/{len(expected)})")
        return 1

    by_model: Dict[str, Dict[Tuple[str, Optional[int], str], dict]] = defaultdict(dict)
    for r in runs:
        by_model[r["label"]][(r["selection"], r["n"], r["variant"])] = r

    # ── so sánh cặp trong từng model ──
    comps = []

    def add_comparison(label: str, fam: str, run_a: dict, run_b: dict) -> None:
        cmp = compare_runs(run_a["samples"], run_b["samples"])
        for metric in ("plus", "base"):
            c = cmp[metric]
            n_common = cmp["n_common"]
            b_cnt, c_cnt = c["a_fail_b_pass"], c["a_pass_b_fail"]
            se = math.sqrt(max(0.0, (b_cnt + c_cnt) - (b_cnt - c_cnt) ** 2 / max(1, n_common))) / max(1, n_common) * 100
            comps.append({
                "label": label, "family": fam, "metric": metric, "n_common": n_common,
                "a": c["a_pass@1"], "b": c["b_pass@1"], "delta_pp": c["delta_pp"],
                "ci95_low": round(c["delta_pp"] - 1.96 * se, 2), "ci95_high": round(c["delta_pp"] + 1.96 * se, 2),
                "a_fail_b_pass": b_cnt, "a_pass_b_fail": c_cnt, "p": c["mcnemar_p"],
            })

    for label, rr in by_model.items():
        ns = sorted({k[1] for k in rr if k[1] is not None})
        variants = sorted({k[2] for k in rr})
        for n in ns:
            for variant in variants:
                for a_sel, b_sel in FAMILIES:
                    ka = ("baseline", None, "") if a_sel == "baseline" else (a_sel, n, variant)
                    kb = (b_sel, n, variant)
                    if ka not in rr or kb not in rr:
                        continue
                    add_comparison(label, f"{a_sel} -> {b_sel} (n={n}{', ' + variant if variant else ''})", rr[ka], rr[kb])
            # giữa các KB / variant: cùng cấu hình retrieved, A < B theo tên variant
            rv = sorted(k[2] for k in rr if k[0] == "retrieved" and k[1] == n)
            for i, va in enumerate(rv):
                for vb in rv[i + 1:]:
                    add_comparison(label, f"retrieved: {va or 'mặc định'} -> {vb} (n={n})",
                                   rr[("retrieved", n, va)], rr[("retrieved", n, vb)])
    # giữa hai model, cùng cấu hình (vd trước / sau GRPO): --model_pairs A:B
    for pair in args.model_pairs or []:
        a_label, _, b_label = pair.partition(":")
        if a_label not in by_model or b_label not in by_model:
            print(f"Bỏ qua --model_pairs {pair}: chưa có lần chạy dùng được của {a_label if a_label not in by_model else b_label}")
            continue
        for key in sorted(set(by_model[a_label]) & set(by_model[b_label]), key=lambda k: (SEL_ORDER.index(k[0])
                          if k[0] in SEL_ORDER else len(SEL_ORDER), k[1] or 0, k[2])):
            sel, n, variant = key
            desc = sel + (f", n={n}" if n else "") + (f", {variant}" if variant else "")
            add_comparison(f"{a_label} -> {b_label}", f"model: {a_label} -> {b_label} ({desc})",
                           by_model[a_label][key], by_model[b_label][key])
    # Holm trong từng họ so sánh (cùng family + metric, qua các model)
    groups = defaultdict(list)
    for i, c in enumerate(comps):
        groups[(c["family"], c["metric"])].append(i)
    for idxs in groups.values():
        for i, adj in zip(idxs, holm([comps[i]["p"] for i in idxs])):
            comps[i]["p_holm"] = round(adj, 4)

    # ── ghi CSV ──
    with (out_dir / "matrix_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "provider", "model", "tier", "params_b", "run", "selection", "n", "variant", "base_pass@1", "plus_pass@1"])
        for r in sorted(runs, key=lambda r: (r["label"], r["run"])):
            m = meta.get(r["label"], {})
            w.writerow([r["label"], m.get("provider"), m.get("model"), m.get("tier"), m.get("params_b"), r["run"],
                        r["selection"], r["n"], r["variant"], r["base"], r["plus"]])
    with (out_dir / "matrix_comparisons.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = ["label", "family", "metric", "n_common", "a", "b", "delta_pp", "ci95_low", "ci95_high",
                "a_fail_b_pass", "a_pass_b_fail", "p", "p_holm"]
        w = csv.DictWriter(fh, fieldnames=["tier"] + keys)
        w.writeheader()
        for c in comps:
            w.writerow({"tier": meta.get(c["label"], {}).get("tier"), **c})

    # ── bảng markdown ──
    def model_key(label):
        base = by_model[label].get(("baseline", None, ""))
        return -(base["plus"] if base else -1)

    labels = sorted(by_model, key=model_key)
    # cột = (cấu hình, variant); nếu mọi lần chạy có KB dùng chung một variant thì bỏ variant khỏi tên cột
    combos = sorted({(k[0], k[2]) for rr in by_model.values() for k in rr if k[0] != "baseline"},
                    key=lambda c: (SEL_ORDER.index(c[0]) if c[0] in SEL_ORDER else len(SEL_ORDER), c[1]))
    shared_variant = combos[0][1] if combos and len({v for _, v in combos}) == 1 else None
    columns = ([("baseline", "")] if any(("baseline", None, "") in rr for rr in by_model.values()) else []) + combos
    lines = ["# Kết quả ma trận model × cấu hình (MBPP+, EvalPlus, đủ 378 bài)", "",
             "pass@1 dạng `base / plus` (%). Δ = plus của cấu hình trừ plus của baseline cùng model; "
             "p = McNemar exact, p_Holm hiệu chỉnh qua các model. 1 bài = 0.26 điểm %.", ""]
    if shared_variant:
        lines += [f"Mọi cấu hình có KB dùng variant `{shared_variant}` ({describe_variant(shared_variant)}).", ""]
    header = ["Model", "Nhóm", "Tham số (B)"] + [
        s if (s == "baseline" or shared_variant is not None) else f"{s} [{describe_variant(v)}]" for s, v in columns]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label in labels:
        m = meta.get(label, {})
        cells = [label, str(m.get("tier") or "—"), str(m.get("params_b") or "—")]
        for s, v in columns:
            hits = [r for k, r in by_model[label].items() if k[0] == s and k[2] == v]
            cells.append(" ; ".join(f"{r['base']:.1f} / {r['plus']:.1f}" + (f" (n={r['n']})" if r["n"] and len(hits) > 1 else "")
                                    for r in sorted(hits, key=lambda r: r["n"] or 0)) or "—")
        lines.append("| " + " | ".join(cells) + " |")

    plus_comps = [c for c in comps if c["metric"] == "plus"]
    if plus_comps:
        lines += ["", "## So sánh cặp (plus)", "",
                  "| Model | So sánh | A | B | Δ (điểm %) | KTC 95% | A✗B✓ | A✓B✗ | p | p_Holm |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(plus_comps, key=lambda c: (c["family"], model_key(c["label"]))):
            lines.append(f"| {c['label']} | {c['family']} | {c['a']:.1f} | {c['b']:.1f} | {c['delta_pp']:+.2f} | "
                         f"[{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] | {c['a_fail_b_pass']} | {c['a_pass_b_fail']} | "
                         f"{fmt_p(c['p'])} | {fmt_p(c.get('p_holm'))} |")

    # ── câu hỏi chính: Δ có phụ thuộc độ mạnh của model? ──
    lines += ["", "## KB giúp model mạnh hay yếu nhiều hơn?", ""]
    plot_series = defaultdict(list)
    for fam in sorted({c["family"] for c in plus_comps if c["family"].startswith("baseline ->")}):
        pts = []
        for c in plus_comps:
            if c["family"] != fam:
                continue
            base_run = by_model[c["label"]].get(("baseline", None, ""))
            pts.append((base_run["plus"], c["delta_pp"], c["label"], c))
        if not pts:
            continue
        plot_series[fam] = pts
        rho, p_rho = spearman([p[0] for p in pts], [p[1] for p in pts])
        tiers = defaultdict(list)
        for x, d, label, _ in pts:
            tiers[meta.get(label, {}).get("tier") or "—"].append(d)
        sig_up = [p[2] for p in pts if p[3].get("p_holm", 1) < 0.05 and p[1] > 0]
        sig_down = [p[2] for p in pts if p[3].get("p_holm", 1) < 0.05 and p[1] < 0]
        lines.append(f"**{fam}** — {len(pts)} model; Δ tăng {sum(1 for p in pts if p[1] > 0)}, giảm {sum(1 for p in pts if p[1] < 0)}, "
                     f"bằng {sum(1 for p in pts if p[1] == 0)}.")
        lines.append("- Δ trung bình theo nhóm: " + "; ".join(
            f"{t}: {sum(v) / len(v):+.2f} ({len(v)} model)" for t, v in sorted(tiers.items())))
        if rho is not None:
            lines.append(f"- Tương quan hạng Spearman giữa plus của baseline và Δ: ρ = {rho:+.2f} (p hoán vị = {fmt_p(p_rho)}). "
                         "ρ âm nghĩa là model càng yếu càng được lợi.")
        lines.append(f"- Có ý nghĩa sau Holm: tăng {', '.join(sig_up) or 'không có'}; giảm {', '.join(sig_down) or 'không có'}.")
        lines.append("")

    sim_rows, sim_lines = similarity_strata(by_model, args.near_dup)
    lines += sim_lines
    with (out_dir / "matrix_similarity.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = ["n", "variant", "label", "stratum", "n_tasks", "baseline", "retrieved", "delta_retrieved", "random", "delta_random"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in sim_rows:
            w.writerow(r)

    if incomplete:
        lines += ["## Lần chạy chưa dùng được", ""]
        lines += [f"- {r['label']}/{r['run']}: {r['why']} ({r['n_done']}/{len(expected)} bài)" for r in incomplete]
    (out_dir / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── biểu đồ ──
    png = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fams = [f for f in plot_series if "-> retrieved" in f] or list(plot_series)
        if fams:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            tier_names = sorted({meta.get(p[2], {}).get("tier") or "—" for f in fams for p in plot_series[f]})
            colors = dict(zip(tier_names, ["#2a6fdb", "#e07b39", "#3a9e5f", "#8e44ad", "#7f8c8d"]))
            for f in fams[:1]:
                for x, d, label, c in plot_series[f]:
                    tier = meta.get(label, {}).get("tier") or "—"
                    ax.errorbar(x, d, yerr=[[d - c["ci95_low"]], [c["ci95_high"] - d]], fmt="o", color=colors[tier],
                                ecolor=colors[tier], alpha=0.85, capsize=3)
                    ax.annotate(label, (x, d), textcoords="offset points", xytext=(5, 4), fontsize=8)
                ax.set_title(f"Δ plus pass@1 ({f}) so với độ mạnh của model")
            for t, col in colors.items():
                ax.scatter([], [], color=col, label=t)
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.margins(x=0.12, y=0.1)
            ax.set_xlabel("plus pass@1 của baseline (%)")
            ax.set_ylabel("Δ plus (điểm %), KTC 95%")
            ax.legend(title="Nhóm", fontsize=8)
            fig.tight_layout()
            png = out_dir / "matrix_delta.png"
            fig.savefig(png, dpi=150)
            plt.close(fig)
    except ImportError:
        pass

    print((out_dir / "matrix_summary.md").read_text(encoding="utf-8"))
    print(f"Đã ghi {out_dir}/matrix_runs.csv, matrix_comparisons.csv, matrix_similarity.csv, matrix_summary.md"
          + (f", {png.name}" if png else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

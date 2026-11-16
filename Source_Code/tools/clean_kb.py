"""Kiểm lại một knowledge base bằng harness mới của pipeline và ghi bản đã lọc (KB gốc không bị sửa).

Chạy từ thư mục Source_Code (script thực thi code do LLM sinh — harness đã cô lập tiến trình con: `python -I`,
env sạch, giới hạn CPU/RAM, chặn mạng ở mức Python; vẫn nên chạy trong Docker/VM/Kaggle nếu được):

    python tools/clean_kb.py ../Data/knowledge_base/all_models/knowledge_base.jsonl \
        --out ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl

Một mục được giữ khi:
- lời giải nạp được như module (khối `if __name__ == "__main__":` không chạy, `sys.exit` = trượt),
- mọi dòng test chạy được, có ít nhất một assert và mọi assert đúng (so sánh trần được đổi thành assert),
- test thật sự gọi tới hàm/lớp do lời giải định nghĩa,
- đề không trùng (sau chuẩn hoá) với mục đã giữ trước đó,
- [--decontam_threshold] đề không quá giống một bài MBPP+ (cosine TF-IDF, mặc định 0.5; 0 = tắt),
- [--drop_test_repaired] test chưa từng bị sửa cho khớp lời giải.
Báo cáo từng loại ghi vào <out>.report.json.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent0.core.config import DEFAULT_DATA_FILE  # noqa: E402
from agent0.data.mbpp_loader import load_mbpp_plus_ids, normalize_test_lines  # noqa: E402
from agent0.data.text_similarity import build_mbpp_plus_index, prompt_key  # noqa: E402
from agent0.execution.harness import run_harness  # noqa: E402


def verdict(row: dict, timeout: float) -> str:
    sol = row.get("solution_code") or ""
    raw = list(row.get("test_list") or []) + list(row.get("challenge_test_list") or [])
    tests, n_assert = normalize_test_lines(raw)
    if not sol.strip():
        return "no_solution"
    if n_assert == 0:
        return "no_assert_in_tests"
    res = run_harness(sol, tests, row.get("test_setup_code") or "", timeout=timeout)
    if res.get("timed_out"):
        return "timeout"
    if not res.get("loaded"):
        return "solution_fails_to_load"
    if res.get("syntax_errors") or res.get("statement_errors"):
        return "test_line_errors"
    if not res.get("referenced"):
        return "tests_do_not_call_solution"
    if res.get("passed"):
        return "verified_with_converted_lines" if res.get("converted") else "verified"
    return "an_assert_fails"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kb", help="knowledge_base.jsonl cần kiểm")
    ap.add_argument("--out", required=True, help="File KB đã lọc (không được trùng file gốc)")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--decontam_threshold", type=float, default=0.5)
    ap.add_argument("--drop_test_repaired", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.kb).resolve(), Path(args.out).resolve()
    if src == out:
        raise SystemExit("--out phải khác file gốc")
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        verdicts = list(pool.map(lambda r: verdict(r, args.timeout), rows))

    decontam = None
    if args.decontam_threshold > 0:
        decontam = build_mbpp_plus_index(DEFAULT_DATA_FILE, load_mbpp_plus_ids())

    kept, seen = [], set()
    reasons: Counter = Counter()
    per_model: dict = defaultdict(Counter)
    examples: dict = defaultdict(list)
    for row, v in zip(rows, verdicts):
        reason = None
        prompt = row.get("original_prompt") or row.get("curriculum_prompt") or ""
        key = prompt_key(prompt)
        if not v.startswith("verified"):
            reason = v
        elif key in seen:
            reason = "duplicate_prompt"
        elif args.drop_test_repaired and (row.get("tests_modified") or row.get("test_repair_rounds_used")):
            reason = "tests_were_repaired"
        elif decontam is not None:
            sim, plus_id, _ = decontam.most_similar(prompt)
            if sim >= args.decontam_threshold:
                reason = "similar_to_mbpp_plus"
                if len(examples[reason]) < 10:
                    examples[reason].append({"prompt": prompt[:120], "mbpp_plus": plus_id, "cosine": round(sim, 3)})
        model = row.get("source_model", "?")
        if reason is None:
            kept.append(row)
            seen.add(key)
            reasons["kept"] += 1
            per_model[model]["kept"] += 1
        else:
            reasons[reason] += 1
            per_model[model][reason] += 1
            if reason != "similar_to_mbpp_plus" and len(examples[reason]) < 5:
                examples[reason].append({"uid": row.get("uid"), "prompt": prompt[:120]})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    report = {
        "source": os.path.relpath(src), "output": os.path.relpath(out), "rows": len(rows), "kept": len(kept),
        "harness_verdicts": dict(Counter(verdicts)), "decisions": dict(reasons),
        "per_model": {m: dict(c) for m, c in per_model.items()}, "examples": examples,
        "options": {"timeout": args.timeout, "decontam_threshold": args.decontam_threshold,
                    "drop_test_repaired": args.drop_test_repaired},
    }
    out.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{src.name}: {len(rows)} mục -> giữ {len(kept)} ({len(kept) / max(1, len(rows)):.1%})")
    for k, v in reasons.most_common():
        print(f"  {k:32s} {v}")
    print(f"Đã ghi {out} và {out.with_suffix('.report.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Dữ liệu RL cho executor: mỗi đề trong KB thành một prompt giống hệt prompt MBPP+ của benchmark.

Prompt benchmark (run_benchmark.build_messages, kiểu evalplus) là docstring gồm mô tả đề và MỘT assert ví dụ.
Đề KB được viết lại đúng dạng đó, nên những gì model học khi RL dùng được ngay lúc đánh giá MBPP+.
Thưởng được chấm trên MỌI test của đề (test_list + challenge_test_list), giống base + plus test của MBPP+.

Không đưa bài MBPP+ vào dữ liệu huấn luyện: bỏ mục KB giống một bài MBPP+ (cosine TF-IDF >= ngưỡng).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from agent0.retrieval.retriever import assert_example_lines


def read_jsonl(path) -> List[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows: Iterable[dict]) -> None:
    """Ghi qua file tạm rồi đổi tên: tiến trình khác đang đọc không bao giờ thấy file dở."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def kb_tests(entry: dict) -> List[str]:
    """Mọi test của đề, đã chuẩn hoá như pipeline (so sánh trần -> assert, bỏ dòng chỉ là literal)."""
    from agent0.data.mbpp_loader import normalize_test_lines

    raw = [t for t in list(entry.get("test_list") or []) + list(entry.get("challenge_test_list") or [])
           if isinstance(t, str) and t.strip()]
    return normalize_test_lines(raw)[0]


def count_asserts(tests: Sequence[str]) -> int:
    from agent0.data.mbpp_loader import normalize_test_lines

    return normalize_test_lines(list(tests))[1]


def example_lines(entry: dict) -> List[str]:
    """Các dòng đưa vào prompt: tới hết assert đầu tiên của test_list (test nhiều dòng được tách dòng)."""
    from agent0.data.mbpp_loader import normalize_test_lines

    tests = normalize_test_lines([t for t in (entry.get("test_list") or []) if isinstance(t, str)])[0]
    flat = [line for t in tests for line in t.splitlines() if line.strip()]
    return assert_example_lines(flat, n_asserts=1)


def is_test_repaired(entry: dict) -> bool:
    """Giống KnowledgeRetriever: test từng bị sửa cho khớp lời giải (LLM cũ hoặc đồng thuận)."""
    return bool(entry.get("tests_modified")) or bool(entry.get("test_repair_rounds_used"))


def kb_description(entry: dict) -> str:
    return (entry.get("original_prompt") or entry.get("prompt") or "").strip()


def kb_task_prompt(entry: dict) -> str:
    """Docstring kiểu MBPP+: mô tả + assert ví dụ đầu tiên (cùng cách few-shot định dạng 'tests')."""
    return '"""\n' + "\n".join([kb_description(entry)] + example_lines(entry)) + '\n"""\n'


def task_messages(task_prompt: str, prompt_style: str = "evalplus") -> List[dict]:
    from agent0.scripts.run_benchmark import build_messages

    return build_messages({"prompt": task_prompt}, "", prompt_style)


def _uid(entry: dict) -> str:
    """uid trong KB gộp có thể trùng giữa các model (vd easy_medium_hard-synllm_2-1), nên thêm hash của mô tả."""
    digest = hashlib.sha256(kb_description(entry).encode("utf-8")).hexdigest()[:8]
    return f"{entry.get('uid') or 'kb'}#{digest}"


def load_rl_tasks(
    kb_paths: Sequence,
    exclude_test_repaired: bool = True,
    min_asserts: int = 3,
    decontam_threshold: float = 0.5,
    prompt_style: str = "evalplus",
) -> Tuple[List[dict], dict]:
    """Đọc các KB, lọc, khử trùng, khử ô nhiễm với MBPP+. Trả về (các đề, thống kê)."""
    from agent0.data.text_similarity import prompt_key

    stats: Counter = Counter()
    rows: List[dict] = []
    seen = set()
    for kb_path in kb_paths:
        for entry in read_jsonl(kb_path):
            stats["read"] += 1
            desc = kb_description(entry)
            tests = kb_tests(entry)
            if not desc or not tests or not example_lines(entry):
                stats["invalid"] += 1
                continue
            if exclude_test_repaired and is_test_repaired(entry):
                stats["test_repaired"] += 1
                continue
            if count_asserts(tests) < min_asserts:
                stats["few_asserts"] += 1
                continue
            key = prompt_key(desc)
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            task_prompt = kb_task_prompt(entry)
            rows.append({
                "uid": _uid(entry),
                "kb_uid": entry.get("uid"),
                "source_kb": str(kb_path),
                "source_model": entry.get("source_model"),
                "description": desc,
                "task_prompt": task_prompt,
                "prompt": task_messages(task_prompt, prompt_style),
                "tests": tests,
                "setup": entry.get("test_setup_code") or "",
                "kb_p_hat": entry.get("p_hat"),
            })
    if decontam_threshold and decontam_threshold > 0:
        rows, n_cont, worst = decontaminate(rows, decontam_threshold)
        stats["contaminated"] = n_cont
        stats["max_similarity_kept"] = worst
    stats["kept"] = len(rows)
    return rows, dict(stats)


def decontaminate(rows: List[dict], threshold: float) -> Tuple[List[dict], int, float]:
    """Bỏ đề giống một bài MBPP+ (cosine TF-IDF >= threshold), như bộ lọc của run_curriculum."""
    from agent0.core.config import DEFAULT_DATA_FILE
    from agent0.data.mbpp_loader import load_mbpp_plus_ids
    from agent0.data.text_similarity import build_mbpp_plus_index

    index = build_mbpp_plus_index(DEFAULT_DATA_FILE, load_mbpp_plus_ids())
    kept, removed, worst = [], 0, 0.0
    for row in rows:
        sim, plus_id, _ = index.most_similar(row["description"])
        if sim >= threshold:
            removed += 1
            continue
        row["mbpp_plus_similarity"] = round(float(sim), 4)
        row["mbpp_plus_nearest"] = plus_id
        worst = max(worst, float(sim))
        kept.append(row)
    return kept, removed, round(worst, 4)


def split_train_val(rows: List[dict], val_frac: float = 0.1, seed: int = 0) -> Tuple[List[dict], List[dict]]:
    """Chia cố định theo hash(uid, seed): thêm KB mới không làm xáo trộn tập val của các đề cũ."""
    train, val = [], []
    for row in rows:
        h = int(hashlib.sha256(f"{seed}:{row['uid']}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if h < val_frac else train).append(row)
    return train, val


def frontier_filter(
    rows: List[dict], p_hats: Dict[str, float], band: Tuple[float, float] = (0.25, 0.75), min_rows: int = 64,
) -> Tuple[List[dict], str]:
    """Giữ đề ở biên năng lực của chính model (Agent0: |p_hat - 0.5| <= 0.25).

    Quá ít đề thì nới dần: 0 < p_hat < 1, rồi p_hat < 1. Trả về (các đề, quy tắc đã dùng)."""
    low, high = band
    rules = [
        (f"{low} <= p_hat <= {high}", lambda p: low <= p <= high),
        ("0 < p_hat < 1", lambda p: 0.0 < p < 1.0),
        ("p_hat < 1", lambda p: p < 1.0),
    ]
    chosen: List[dict] = []
    for name, keep in rules:
        chosen = [r for r in rows if r["uid"] in p_hats and keep(p_hats[r["uid"]])]
        if len(chosen) >= min_rows:
            return chosen, name
    return chosen, rules[-1][0] + f" (chỉ {len(chosen)} đề < {min_rows})"

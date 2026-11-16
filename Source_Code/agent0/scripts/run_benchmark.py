#!/usr/bin/env python3
"""Benchmark MBPP+ (EvalPlus) cho baseline và few-shot RAG từ knowledge base.

Các lệnh con (chạy từ thư mục Source_Code: `python -m agent0.scripts.run_benchmark <lệnh> ...`):

  generate  Sinh lời giải cho các bài MBPP+. Ghi từng dòng ngay khi có kết quả; lỗi API được thử lại
            với backoff, nếu vẫn lỗi thì ghi vào <run>_errors.jsonl và KHÔNG ghi gì vào samples.
            --resume chỉ chạy các bài còn thiếu.
  check     Kiểm tra file samples: đủ 378 bài, không có stub, không trùng task_id.
  evaluate  Chấm bằng EvalPlus (trên máy hoặc trong Docker), đọc <run>_eval_results.json
            -> pass@1 base và plus trên đủ 378 bài. Từ chối chấm file thiếu bài / có stub.
  run       baseline + with_kb -> evaluate -> so sánh cặp -> benchmark_summary.json.
  compare   So sánh cặp hai lần chạy (số bài lật hai chiều + McNemar exact).
  dev       Chọn số ví dụ n trên tập dev (các bài MBPP KHÔNG có trong MBPP+), không đụng tới MBPP+.

Few-shot (chế độ with_kb):
  --example_selection retrieved|random|fixed  ví dụ truy hồi theo độ giống / ngẫu nhiên / cố định cho mọi bài
                                              (ablation: tách tác dụng "truy hồi" khỏi tác dụng "có ví dụ").
  --fewshot_format tests|full|code           tests = đề + assert ví dụ + lời giải (cùng định dạng prompt MBPP+).
  --min_similarity τ                         ví dụ dưới τ bị bỏ; bài không còn ví dụ nào chạy zero-shot.
  Mục KB sinh từ chính các bài đang đánh giá (theo source_task_id hoặc trùng đề) luôn bị loại khỏi KB khi truy hồi.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from agent0.core.config import (
    DEFAULT_DATA_FILE,
    DEFAULT_DEV_RESULTS_DIR,
    DEFAULT_KB_PATH,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RESULTS_DIR,
    MBPP_PLUS_TASK_IDS_FILE,
    REPO_ROOT,
    SCRIPT_DIR,
    SUPPORTED_PROVIDERS,
    RuntimeConfig,
    describe_runtime,
    resolve_runtime,
)
from agent0.core.llm_client import QuotaExhaustedError, call_stats, chat_completion, configure_cache
from agent0.core.logger import setup_logger
from agent0.core.utils import (
    api_key_from_args,
    extract_code_block,
    parse_extra_body,
    sanitized_argv,
    strip_think_blocks,
)

logger = logging.getLogger("agent0")

EVALPLUS_VERSION = "0.3.1"
DOCKER_IMAGE = f"agent0-evalplus:{EVALPLUS_VERSION}"
DOCKERFILE = SCRIPT_DIR / "tools" / "evalplus.Dockerfile"
DOCKER_CACHE_VOLUME = "agent0-evalplus-cache"

STUB_RE = re.compile(r"^\s*def\s+\w+\s*\(\s*\*args\s*,\s*\*\*kwargs\s*\)\s*:\s*pass\s*$")
EVALPLUS_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
SYSTEM_PROMPT = "You are a Python coding assistant."
LEGACY_SYSTEM_PROMPT = "You are a Python coding assistant. Return only runnable Python code, no markdown and no explanation."
PROMPT_STYLES = ("evalplus", "with_tests")

MACOS_WARNING = (
    "Đang chấm EvalPlus trực tiếp trên macOS: các lần chấm trước trên Mac bị timeout hàng loạt "
    "(MBPP+ thấp hơn thực tế 17–20 điểm). Nên dùng --evaluator docker hoặc chấm trên Linux/Kaggle."
)


class GenerationError(RuntimeError):
    pass


class IncompleteSamplesError(RuntimeError):
    pass


class EvaluationError(RuntimeError):
    pass


# ───────────────────────────── tiện ích chung ─────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def run_name(mode: str, n_examples: int, tag: Optional[str] = None, prefix: str = "", variant: str = "") -> str:
    name = "baseline" if mode == "baseline" else f"with_kb_n{n_examples}{variant}"
    name = f"{prefix}{name}"
    return f"{name}_{tag}" if tag else name


def fewshot_variant(args) -> str:
    """Hậu tố tên lần chạy cho các lựa chọn few-shot khác mặc định, vd _random, _full, _tau0.6, _mmr0.7."""
    parts = []
    if getattr(args, "example_selection", "retrieved") != "retrieved":
        parts.append(args.example_selection)
    if getattr(args, "fewshot_format", "tests") != "tests":
        parts.append(args.fewshot_format)
    if getattr(args, "fewshot_asserts", 1) != 1:
        parts.append(f"a{args.fewshot_asserts}")
    if getattr(args, "min_similarity", 0.0) > 0:
        parts.append(f"tau{args.min_similarity:g}")
    if getattr(args, "mmr_lambda", 1.0) < 1.0:
        parts.append(f"mmr{args.mmr_lambda:g}")
    if getattr(args, "retrieval_query", "description") != "description":
        parts.append(f"q{args.retrieval_query}")
    if getattr(args, "exclude_test_repaired", False):
        parts.append("notr")
    return "".join(f"_{p}" for p in parts)


def sidecar(samples_path, suffix: str) -> Path:
    """File đi kèm: <stem>_errors.jsonl, <stem>_meta.json, <stem>_metrics.json, <stem>_eval_results.json."""
    s = str(samples_path)
    stem = s[: -len(".jsonl")] if s.endswith(".jsonl") else s
    return Path(f"{stem}_{suffix}")


def read_jsonl(path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_stub(solution: Optional[str]) -> bool:
    return not (solution or "").strip() or bool(STUB_RE.match(solution or ""))


def sha256_file(path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint() -> str:
    """Hash toàn bộ mã nguồn package agent0 (thay cho git hash khi repo tải về dạng zip)."""
    h = hashlib.sha256()
    for py in sorted((SCRIPT_DIR / "agent0").rglob("*.py")):
        h.update(str(py.relative_to(SCRIPT_DIR)).encode())
        h.update(py.read_bytes())
    return h.hexdigest()[:16]


def git_commit() -> Optional[str]:
    if not (REPO_ROOT / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_sha256": code_fingerprint(),
        "git_commit": git_commit(),
    }
    try:
        import evalplus  # noqa: F401

        info["evalplus_local"] = getattr(evalplus, "__version__", "unknown")
    except Exception:
        info["evalplus_local"] = None
    return info


def kb_info(kb_path) -> Optional[dict]:
    if not kb_path:
        return None
    p = Path(kb_path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    with p.open(encoding="utf-8") as fh:
        n = sum(1 for line in fh if line.strip())
    return {"path": str(p), "exists": True, "entries": n, "sha256": sha256_file(p)}


# ───────────────────────────── dữ liệu ─────────────────────────────

def load_expected_task_ids(path=MBPP_PLUS_TASK_IDS_FILE) -> List[str]:
    """378 task_id của MBPP+ v0.2.0 (file tĩnh, không cần cài evalplus)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [t["task_id"] for t in data["tasks"]]


def load_mbpp_plus_tasks() -> List[dict]:
    """Tải 378 bài MBPP+ từ EvalPlus (cần `pip install evalplus`)."""
    from evalplus.data import get_mbpp_plus

    tasks = []
    for task_id, data in get_mbpp_plus().items():
        assertion = data.get("assertion", "")
        base_tests = [line.strip() for line in assertion.split("\n") if line.strip().startswith("assert")]
        tasks.append({
            "task_id": task_id,
            "prompt": data.get("prompt", ""),
            "entry_point": data.get("entry_point", "solution"),
            # Chỉ dùng khi --prompt_style with_tests (giao thức cũ, đưa base test vào prompt)
            "test_list": base_tests[:5],
        })
    return tasks


def load_dev_tasks(
    data_file=DEFAULT_DATA_FILE,
    dev_size: Optional[int] = 200,
    dev_seed: int = 0,
    exclude_ids: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Tập dev: các bài trong MBPP gốc KHÔNG có trong MBPP+ (596 bài), lấy mẫu cố định theo dev_seed."""
    exclude = {int(t.split("/")[1]) for t in (exclude_ids or load_expected_task_ids())}
    rows = [r for r in read_jsonl(data_file) if int(r["task_id"]) not in exclude]
    rows.sort(key=lambda r: int(r["task_id"]))
    if dev_size and dev_size < len(rows):
        rows = random.Random(dev_seed).sample(rows, dev_size)
        rows.sort(key=lambda r: int(r["task_id"]))
    tasks = []
    for r in rows:
        tests = [t for t in r.get("test_list", []) if t.strip()]
        if not tests:
            continue
        m = re.search(r"^\s*def\s+(\w+)\s*\(", r.get("code", ""), re.M)
        tasks.append({
            "task_id": f"MbppDev/{r['task_id']}",
            # cùng định dạng với prompt MBPP+: mô tả + 1 assert ví dụ trong docstring
            "prompt": f'"""\n{r["text"].strip()}\n{tests[0].strip()}\n"""\n',
            "entry_point": m.group(1) if m else "solution",
            "test_list": tests,
            "test_setup_code": r.get("test_setup_code", "") or "",
        })
    return tasks


# ───────────────────────────── sinh lời giải ─────────────────────────────

def build_messages(task: dict, few_shot: str = "", prompt_style: str = "evalplus") -> List[dict]:
    if prompt_style == "with_tests":
        # Giao thức cũ của dự án: đưa thêm base test vào prompt (không so được với leaderboard EvalPlus).
        tests = json.dumps(task.get("test_list", [])[:5], indent=2)
        if few_shot:
            user = (
                f"Here are similar solved tasks for reference:\n\n{few_shot}\n\n"
                f"Now solve the new task below following the same patterns.\n\nTask:\n{task['prompt']}\n\n"
                f"Tests:\n{tests}\n\nReturn only the Python solution code."
            )
        else:
            user = f"Task:\n{task['prompt']}\n\nTests:\n{tests}\n\nReturn only the Python solution code."
        return [{"role": "system", "content": LEGACY_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if prompt_style != "evalplus":
        raise ValueError(f"prompt_style không hợp lệ: {prompt_style}")
    # Mặc định: chỉ prompt gốc của bài (có sẵn 1 assert ví dụ), theo hướng dẫn chat của EvalPlus.
    problem = f"{EVALPLUS_INSTRUCTION}\n```\n{task['prompt'].strip()}\n```"
    user = f"Here are some solved example tasks for reference:\n\n{few_shot}\n\n{problem}" if few_shot else problem
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_solution(generated: str) -> str:
    text = strip_think_blocks(generated or "")
    code = extract_code_block(text)
    return (code if code is not None else text).strip()


def task_description(prompt: str) -> str:
    """Mô tả của bài (bỏ dấu docstring và các dòng assert) — dùng làm câu truy vấn và để loại đề trùng khỏi KB."""
    text = (prompt or "").strip().strip('"').strip()
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("assert")).strip()


class FewShotSelector:
    """Chọn ví dụ few-shot cho một bài: truy hồi theo độ giống, ngẫu nhiên, hoặc một bộ cố định."""

    SELECTIONS = ("retrieved", "random", "fixed")

    def __init__(
        self,
        retriever,
        n_examples: int,
        selection: str = "retrieved",
        fewshot_format: str = "tests",
        n_asserts: int = 1,
        min_similarity: float = 0.0,
        mmr_lambda: float = 1.0,
        retrieval_query: str = "description",
        seed: int = 0,
    ):
        if selection not in self.SELECTIONS:
            raise ValueError(f"example_selection không hợp lệ: {selection}")
        self.retriever = retriever
        self.n = n_examples
        self.selection = selection
        self.fewshot_format = fewshot_format
        self.n_asserts = n_asserts
        self.min_similarity = min_similarity
        self.mmr_lambda = mmr_lambda
        self.retrieval_query = retrieval_query
        self.seed = seed
        self._fixed = retriever.sample(n_examples, random.Random(f"fixed:{seed}")) if selection == "fixed" else None

    def select(self, task: dict) -> Tuple[str, dict]:
        if self.selection == "retrieved":
            query = task_description(task["prompt"]) if self.retrieval_query == "description" else task["prompt"]
            examples = self.retriever.query(query, n=self.n, min_similarity=self.min_similarity, mmr_lambda=self.mmr_lambda)
        elif self.selection == "random":
            examples = self.retriever.sample(self.n, random.Random(f"random:{self.seed}:{task['task_id']}"))
        else:
            examples = list(self._fixed or [])
        text = self.retriever.format_few_shot(examples, style=self.fewshot_format, n_asserts=self.n_asserts)
        info = {
            "n": len(examples),
            "uids": [e.get("uid") for e in examples],
            "similarities": [e.get("similarity") for e in examples],
        }
        return text, info

    def describe(self) -> dict:
        return {
            "n_examples": self.n,
            "example_selection": self.selection,
            "fewshot_format": self.fewshot_format,
            "fewshot_asserts": self.n_asserts,
            "min_similarity": self.min_similarity,
            "mmr_lambda": self.mmr_lambda,
            "retrieval_query": self.retrieval_query,
            "fixed_uids": [e.get("uid") for e in self._fixed] if self._fixed is not None else None,
            "kb_filter": dict(getattr(self.retriever, "stats", {}) or {}),
        }


def generate_solution_for_task(
    runtime: RuntimeConfig,
    task: dict,
    selector: Optional[FewShotSelector] = None,
    prompt_style: str = "evalplus",
    temperature: float = 0.0,
    seed: Optional[int] = 0,
    max_tokens: int = 1800,
    max_retries: Optional[int] = None,
) -> Tuple[str, Optional[dict]]:
    """Trả về (lời giải, thông tin ví dụ few-shot hoặc None)."""
    few_shot, info = ("", None)
    if selector is not None and selector.n > 0:
        few_shot, info = selector.select(task)
    messages = build_messages(task, few_shot, prompt_style)
    generated = chat_completion(
        runtime, messages, temperature=temperature, max_tokens=max_tokens, seed=seed, max_retries=max_retries
    )
    solution = extract_solution(generated)
    if not solution:
        raise GenerationError("model trả về rỗng")
    if is_stub(solution):
        raise GenerationError("model trả về stub")
    return solution, info


def _retrieval_summary(infos: List[dict]) -> Optional[dict]:
    if not infos:
        return None
    tops = [i["similarities"][0] for i in infos if i.get("similarities") and i["similarities"][0] is not None]
    return {
        "tasks": len(infos),
        "zero_shot": sum(1 for i in infos if i["n"] == 0),
        "mean_examples": round(sum(i["n"] for i in infos) / len(infos), 3),
        "mean_top_similarity": round(sum(tops) / len(tops), 4) if tops else None,
    }


def generate_all(
    runtime: RuntimeConfig,
    tasks: List[dict],
    output_file,
    selector: Optional[FewShotSelector] = None,
    prompt_style: str = "evalplus",
    temperature: float = 0.0,
    seed: Optional[int] = 0,
    max_tokens: int = 1800,
    max_retries: Optional[int] = None,
    resume: bool = False,
    overwrite: bool = False,
    workers: int = 1,
    max_consecutive_errors: int = 0,
) -> dict:
    """Sinh lời giải, ghi từng dòng. Lỗi -> <run>_errors.jsonl, không bao giờ ghi stub vào samples.

    workers > 1: gọi model song song (I/O-bound); các dòng được ghi theo thứ tự hoàn thành (EvalPlus khớp theo task_id).
    Ví dụ few-shot đã dùng cho từng bài được ghi vào <run>_retrieval.jsonl.
    max_consecutive_errors > 0: dừng sớm khi chừng ấy bài liên tiếp lỗi (hết quota, sai tên model, mất mạng);
    các bài chưa sinh để lại cho lần --resume sau. Hết hạn mức (QuotaExhaustedError) thì dừng ngay.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    errors_file = sidecar(output_file, "errors.jsonl")
    retrieval_file = sidecar(output_file, "retrieval.jsonl")

    kept: Dict[str, dict] = {}
    dropped = 0
    if output_file.exists():
        if resume:
            for row in read_jsonl(output_file):
                tid = row.get("task_id")
                if tid and not is_stub(row.get("solution")) and tid not in kept:
                    kept[tid] = {"task_id": tid, "solution": row["solution"]}
                else:
                    dropped += 1
        elif not overwrite:
            raise FileExistsError(
                f"{output_file} đã tồn tại. Dùng --resume để chạy tiếp các bài còn thiếu, hoặc --overwrite để ghi đè."
            )
        elif retrieval_file.exists():
            retrieval_file.unlink()

    # Ghi lại phần giữ được (bỏ stub / dòng trùng của lần chạy cũ) rồi nối thêm bài mới.
    order = [t["task_id"] for t in tasks]
    with output_file.open("w", encoding="utf-8") as fh:
        for tid in order + [t for t in kept if t not in set(order)]:
            if tid in kept:
                fh.write(json.dumps(kept[tid], ensure_ascii=False) + "\n")

    todo = [t for t in tasks if t["task_id"] not in kept]
    if resume:
        logger.info("Resume: giữ %s bài, bỏ %s dòng stub/trùng, còn %s bài cần sinh", len(kept), dropped, len(todo))

    label = f"(KB n={selector.n}, {selector.selection})" if selector is not None else "(baseline)"

    def work(idx_task):
        idx, task = idx_task
        logger.info("Generating %s/%s | task_id=%s %s", idx, len(todo), task["task_id"], label)
        return generate_solution_for_task(
            runtime, task, selector, prompt_style, temperature, seed, max_tokens, max_retries
        )

    generated = errors = consecutive = 0
    stopped_reason: Optional[str] = None
    infos: List[dict] = []
    ferr = fret = None
    try:
        with output_file.open("a", encoding="utf-8") as fh:

            def record(task, result=None, exc=None):
                """Ghi kết quả một bài; trả về True nếu phải dừng sớm."""
                nonlocal generated, errors, consecutive, stopped_reason, ferr, fret
                if exc is not None:
                    errors += 1
                    consecutive += 1
                    if ferr is None:
                        ferr = errors_file.open("a", encoding="utf-8")
                    ferr.write(json.dumps({
                        "task_id": task["task_id"], "time": now_iso(),
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                    }, ensure_ascii=False) + "\n")
                    ferr.flush()
                    logger.error("Generate FAILED | task_id=%s | %s: %s", task["task_id"], type(exc).__name__, str(exc)[:300])
                    if isinstance(exc, QuotaExhaustedError):
                        stopped_reason = f"hết hạn mức provider: {str(exc)[:300]}"
                    elif max_consecutive_errors and consecutive >= max_consecutive_errors:
                        stopped_reason = f"{consecutive} bài liên tiếp lỗi"
                    return stopped_reason is not None
                consecutive = 0
                solution, info = result
                fh.write(json.dumps({"task_id": task["task_id"], "solution": solution}, ensure_ascii=False) + "\n")
                fh.flush()
                generated += 1
                if info is not None:
                    infos.append(info)
                    if fret is None:
                        fret = retrieval_file.open("a", encoding="utf-8")
                    fret.write(json.dumps({"task_id": task["task_id"], **info}, ensure_ascii=False) + "\n")
                    fret.flush()
                return False

            if workers <= 1:
                for item in enumerate(todo, 1):
                    try:
                        result = work(item)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001 - mọi lỗi đều được ghi lại, không thay bằng stub
                        if record(item[1], exc=exc):
                            break
                        continue
                    record(item[1], result)
            else:
                pool = ThreadPoolExecutor(max_workers=workers)
                try:
                    futures = {pool.submit(work, item): item[1] for item in enumerate(todo, 1)}
                    for fut in as_completed(futures):
                        task = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            if record(task, exc=exc):
                                pool.shutdown(wait=False, cancel_futures=True)
                                break
                            continue
                        record(task, result)
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                pool.shutdown(wait=stopped_reason is None)
    finally:
        for f in (ferr, fret):
            if f is not None:
                f.close()

    stats = {
        "output_file": str(output_file),
        "errors_file": str(errors_file) if errors else None,
        "requested": len(tasks),
        "reused": len([t for t in order if t in kept]),
        "generated": generated,
        "errors": errors,
        "missing": len(tasks) - len([t for t in order if t in kept]) - generated,
        "workers": workers,
        "stopped_early": stopped_reason,
        "retrieval": _retrieval_summary(infos),
    }
    if stopped_reason:
        logger.error("Dừng sớm: %s. Còn %s bài chưa sinh — chạy lại với --resume.", stopped_reason, stats["missing"])
    elif errors:
        logger.warning("%s bài lỗi (xem %s). Chạy lại cùng lệnh với --resume để sinh tiếp.", errors, errors_file)
    return stats


# ───────────────────────────── kiểm tra samples ─────────────────────────────

def check_samples(samples_file, expected_ids: Optional[List[str]] = None) -> dict:
    expected = list(expected_ids) if expected_ids is not None else load_expected_task_ids()
    rows = read_jsonl(samples_file) if Path(samples_file).exists() else []
    counts = Counter(r.get("task_id") for r in rows)
    present = set(counts)
    report = {
        "samples_file": str(samples_file),
        "expected": len(expected),
        "rows": len(rows),
        "missing": [t for t in expected if t not in present],
        "stubs": sorted({r.get("task_id") for r in rows if is_stub(r.get("solution"))}),
        "duplicates": sorted(t for t, c in counts.items() if c > 1),
        "unknown": sorted(t for t in present if t not in set(expected)),
    }
    report["complete"] = not (report["missing"] or report["stubs"] or report["duplicates"] or report["unknown"])
    return report


def format_check_report(rep: dict) -> str:
    def few(items):
        return ", ".join(items[:8]) + (" ..." if len(items) > 8 else "")

    lines = [f"{rep['samples_file']}: {rep['rows']} dòng / {rep['expected']} bài cần có"]
    for key, label in (("missing", "thiếu"), ("stubs", "stub"), ("duplicates", "trùng task_id"), ("unknown", "task_id lạ")):
        if rep[key]:
            lines.append(f"  - {label}: {len(rep[key])} ({few(rep[key])})")
    lines.append("  => ĐỦ, chấm được" if rep["complete"] else "  => CHƯA ĐỦ")
    return "\n".join(lines)


# ───────────────────────────── chấm EvalPlus ─────────────────────────────

def load_eval_verdicts(results_path) -> Dict[str, Tuple[float, float]]:
    """task_id -> (tỉ lệ pass base, tỉ lệ pass plus). Plus = qua cả base lẫn plus (như EvalPlus)."""
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    verdicts = {}
    for tid, rows in data.get("eval", {}).items():
        if not rows:
            continue
        base = sum(r.get("base_status") == "pass" for r in rows) / len(rows)
        plus = sum(r.get("base_status") == "pass" and r.get("plus_status") == "pass" for r in rows) / len(rows)
        verdicts[tid] = (base, plus)
    return verdicts


def compute_metrics(results_path, expected_ids: List[str], attempted_ids: Optional[Iterable[str]] = None) -> dict:
    v = load_eval_verdicts(results_path)
    n = len(expected_ids)
    out = {
        "n_tasks": n,
        "base_pass@1": round(100 * sum(v.get(t, (0.0, 0.0))[0] for t in expected_ids) / n, 2),
        "plus_pass@1": round(100 * sum(v.get(t, (0.0, 0.0))[1] for t in expected_ids) / n, 2),
    }
    if attempted_ids is not None:
        attempted = [t for t in expected_ids if t in set(attempted_ids)]
        if len(attempted) < n and attempted:
            out["attempted_only"] = {
                "n_tasks": len(attempted),
                "base_pass@1": round(100 * sum(v.get(t, (0.0, 0.0))[0] for t in attempted) / len(attempted), 2),
                "plus_pass@1": round(100 * sum(v.get(t, (0.0, 0.0))[1] for t in attempted) / len(attempted), 2),
                "note": "chỉ để gỡ lỗi; không dùng để báo cáo",
            }
    return out


def choose_evaluator(evaluator: str) -> str:
    if evaluator == "auto":
        if sys.platform == "darwin" and shutil.which("docker"):
            return "docker"
        return "local"
    if evaluator not in ("local", "docker"):
        raise ValueError(f"evaluator không hợp lệ: {evaluator}")
    return evaluator


def ensure_docker_image(image: str = DOCKER_IMAGE) -> None:
    if not shutil.which("docker"):
        raise EvaluationError("Không tìm thấy lệnh docker. Cài Docker Desktop hoặc dùng --evaluator local trên Linux.")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0:
        return
    logger.info("Build image %s từ %s (chỉ lần đầu)...", image, DOCKERFILE)
    proc = subprocess.run(
        ["docker", "build", "-t", image, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise EvaluationError(f"docker build thất bại:\n{proc.stderr[-3000:]}")


def run_evalplus(eval_file: Path, evaluator: str, force: bool, parallel: Optional[int] = None) -> None:
    def args_for(samples_path: str) -> List[str]:
        a = ["--dataset", "mbpp", "--samples", samples_path]
        if parallel:
            a += ["--parallel", str(parallel)]
        if force:
            a += ["--i_just_wanna_run"]
        return a

    if evaluator == "local":
        try:
            import evalplus  # noqa: F401
        except ImportError as exc:
            raise EvaluationError("Chưa cài evalplus: pip install evalplus==0.3.1 (hoặc dùng --evaluator docker)") from exc
        cmd = [sys.executable, "-m", "evalplus.evaluate"] + args_for(str(eval_file))
    else:
        ensure_docker_image()
        inner = " ".join(shlex.quote(a) for a in ["evalplus.evaluate"] + args_for(f"/w/{eval_file.name}"))
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{eval_file.parent.resolve()}:/w", "-w", "/w",
            "-v", f"{DOCKER_CACHE_VOLUME}:/root/.cache",
            DOCKER_IMAGE, "bash", "-lc", inner,
        ]
    logger.info("Chạy EvalPlus: %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-6:])
    if tail:
        logger.info("EvalPlus output (cuối):\n%s", tail)
    if proc.returncode != 0:
        raise EvaluationError(f"EvalPlus lỗi (exit {proc.returncode}):\n{(proc.stderr or proc.stdout)[-3000:]}")


def evaluate_samples(
    samples_file,
    evaluator: str = "auto",
    allow_incomplete: bool = False,
    rerun: bool = False,
    parallel: Optional[int] = None,
) -> dict:
    samples = Path(samples_file).resolve()
    expected = load_expected_task_ids()
    report = check_samples(samples, expected)
    if not report["complete"]:
        msg = format_check_report(report)
        if not allow_incomplete:
            raise IncompleteSamplesError(
                msg + "\nChạy lại `generate ... --resume` để sinh các bài còn thiếu, "
                "hoặc thêm --allow_incomplete (bài thiếu/stub tính là fail, kết quả ghi partial)."
            )
        logger.warning("%s\n--allow_incomplete: bài thiếu/stub được tính là fail.", msg)
        # EvalPlus bắt buộc đủ mọi bài -> tạo bản đệm riêng, không sửa file gốc.
        eval_file = sidecar(samples, "padded.jsonl")
        kept = {}
        for row in read_jsonl(samples):
            tid = row.get("task_id")
            if tid in set(expected) and tid not in kept and not is_stub(row.get("solution")):
                kept[tid] = row
        with eval_file.open("w", encoding="utf-8") as fh:
            for tid in expected:
                row = kept.get(tid) or {"task_id": tid, "solution": "# missing\n"}
                fh.write(json.dumps({"task_id": tid, "solution": row["solution"]}, ensure_ascii=False) + "\n")
        attempted = set(kept)
    else:
        eval_file = samples
        attempted = set(expected)

    results_path = sidecar(eval_file, "eval_results.json")
    stale = results_path.exists() and results_path.stat().st_mtime < eval_file.stat().st_mtime
    mode = choose_evaluator(evaluator)
    if mode == "local" and sys.platform == "darwin":
        logger.warning(MACOS_WARNING)
    if stale:
        logger.info("Kết quả chấm cũ hơn file samples -> chấm lại")
    if rerun or stale or not results_path.exists():
        run_evalplus(eval_file, mode, force=bool(rerun or stale), parallel=parallel)
    if not results_path.exists():
        raise EvaluationError(f"EvalPlus không tạo {results_path}")

    metrics = compute_metrics(results_path, expected, attempted)
    metrics.update({
        "samples_file": str(samples),
        "eval_results_file": str(results_path),
        "evaluator": mode,
        "partial": not report["complete"],
        "n_missing": len(report["missing"]),
        "n_stubs": len(report["stubs"]),
        "evaluated_at": now_iso(),
        "environment": environment_info(),
    })
    write_json(sidecar(samples, "metrics.json"), metrics)
    logger.info(
        "pass@1 trên %s bài | base %.2f | plus %.2f%s",
        metrics["n_tasks"], metrics["base_pass@1"], metrics["plus_pass@1"], " (PARTIAL)" if metrics["partial"] else "",
    )
    return metrics


# ───────────────────────────── so sánh cặp ─────────────────────────────

def mcnemar_exact(b: int, c: int) -> float:
    """p-value hai phía của McNemar exact (binomial với p = 0.5 trên các cặp bất đồng)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _results_path_for(path) -> Path:
    """Nhận file samples hoặc file kết quả; với lần chạy partial dùng bản đệm <stem>_padded_eval_results.json."""
    p = Path(path)
    if p.name.endswith("_eval_results.json"):
        return p
    direct = sidecar(p, "eval_results.json")
    padded = sidecar(sidecar(p, "padded.jsonl"), "eval_results.json")
    return direct if direct.exists() or not padded.exists() else padded


def _excluded_ids_for(path) -> set:
    """Các bài không có lời giải thật trong file samples gốc (thiếu hoặc stub) -> loại khỏi so sánh cặp."""
    p = Path(path)
    if p.suffix != ".jsonl":
        stem = str(p)[: -len("_eval_results.json")]
        stem = stem[: -len("_padded")] if stem.endswith("_padded") else stem
        p = Path(stem + ".jsonl")
    if not p.exists():
        return set()
    rows = read_jsonl(p)
    valid = {r.get("task_id") for r in rows if not is_stub(r.get("solution"))}
    try:
        expected = set(load_expected_task_ids())
    except Exception:
        expected = {r.get("task_id") for r in rows}
    return expected - valid


def compare_verdicts(va: Dict[str, Tuple[float, float]], vb: Dict[str, Tuple[float, float]], common: List[str]) -> dict:
    out = {"n_common": len(common)}
    for idx, key in ((0, "base"), (1, "plus")):
        a_pass = [va[t][idx] >= 0.5 for t in common]
        b_pass = [vb[t][idx] >= 0.5 for t in common]
        b_cnt = sum(1 for x, y in zip(a_pass, b_pass) if not x and y)
        c_cnt = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
        n = max(1, len(common))
        out[key] = {
            "a_pass@1": round(100 * sum(a_pass) / n, 2),
            "b_pass@1": round(100 * sum(b_pass) / n, 2),
            "delta_pp": round(100 * (sum(b_pass) - sum(a_pass)) / n, 2),
            "a_fail_b_pass": b_cnt,
            "a_pass_b_fail": c_cnt,
            "mcnemar_p": round(mcnemar_exact(b_cnt, c_cnt), 4),
        }
    out["one_task_pp"] = round(100 / max(1, len(common)), 3)
    return out


def compare_runs(a_path, b_path) -> dict:
    ra, rb = _results_path_for(a_path), _results_path_for(b_path)
    for r in (ra, rb):
        if not r.exists():
            raise FileNotFoundError(f"Chưa có {r} — chạy evaluate trước")
    va, vb = load_eval_verdicts(ra), load_eval_verdicts(rb)
    excluded = _excluded_ids_for(a_path) | _excluded_ids_for(b_path)
    common = sorted((set(va) & set(vb)) - excluded)
    out = compare_verdicts(va, vb, common)
    out.update({"a": str(ra), "b": str(rb), "excluded_missing_or_stub_tasks": len(excluded & (set(va) | set(vb)))})
    return out


def format_comparison(cmp: dict, a_label: str = "A", b_label: str = "B") -> str:
    lines = [f"So sánh cặp trên {cmp['n_common']} bài chung (1 bài = {cmp['one_task_pp']} điểm %)"]
    for key in ("base", "plus"):
        r = cmp[key]
        lines.append(
            f"  {key:4s}: {a_label} {r['a_pass@1']:.2f} -> {b_label} {r['b_pass@1']:.2f} (Δ {r['delta_pp']:+.2f}) | "
            f"{a_label} fail->{b_label} pass {r['a_fail_b_pass']}, {a_label} pass->{b_label} fail {r['a_pass_b_fail']} | "
            f"McNemar p = {r['mcnemar_p']:.4f}"
        )
    return "\n".join(lines)


# ───────────────────────────── retriever / runtime ─────────────────────────────

def build_retriever(
    kb_path,
    embed_model: str,
    embed_base_url: str,
    embed_api_key: Optional[str] = None,
    exclude_tasks: Optional[List[dict]] = None,
    exclude_test_repaired: bool = False,
    need_embeddings: bool = True,
):
    """Nạp KB, loại các mục sinh từ chính các bài đang đánh giá, dựng index embedding (nếu cần)."""
    kb = Path(kb_path)
    if not kb.exists():
        raise FileNotFoundError(f"Không thấy knowledge base: {kb}")
    from agent0.retrieval.retriever import KnowledgeRetriever

    exclude_tasks = exclude_tasks or []
    retriever = KnowledgeRetriever(
        kb_path=str(kb), embed_model=embed_model, embed_base_url=embed_base_url, embed_api_key=embed_api_key,
        exclude_source_task_ids=[t["task_id"] for t in exclude_tasks],
        exclude_prompts=[task_description(t.get("prompt", "")) for t in exclude_tasks],
        exclude_test_repaired=exclude_test_repaired,
    )
    if not need_embeddings:
        retriever.load_entries()
    else:
        try:
            retriever.build_index()
        except Exception as exc:
            raise RuntimeError(
                f"Không dựng được index cho KB ({exc}). Cần Ollama local chạy model embedding "
                f"'{embed_model}' tại {embed_base_url} (ollama pull {embed_model})."
            ) from exc
    removed = {k: v for k, v in retriever.stats.items() if k.startswith("excluded") and v}
    if removed:
        logger.info("KB: loại %s khỏi tập truy hồi", removed)
    if not retriever.entries:
        raise RuntimeError(f"KB {kb} không còn mục nào sau khi lọc ({retriever.stats})")
    return retriever


def build_selector(args, tasks: List[dict], kb_path) -> FewShotSelector:
    retriever = build_retriever(
        kb_path, args.embed_model, args.embed_base_url,
        exclude_tasks=tasks, exclude_test_repaired=args.exclude_test_repaired,
        need_embeddings=args.example_selection == "retrieved",
    )
    return FewShotSelector(
        retriever, args.n_examples, args.example_selection, args.fewshot_format, args.fewshot_asserts,
        args.min_similarity, args.mmr_lambda, args.retrieval_query, args.seed,
    )


def runtime_from_args(args) -> RuntimeConfig:
    runtime = resolve_runtime(args.provider, args.model, args.base_url, api_key_from_args(args))
    if runtime.provider == "offline":
        raise SystemExit("Provider 'offline' không gọi LLM nên không sinh lời giải benchmark được.")
    runtime.extra_body = parse_extra_body(getattr(args, "extra_body", None))
    logger.info("Runtime: %s%s", describe_runtime(runtime),
                f" | extra_body={json.dumps(runtime.extra_body)}" if runtime.extra_body else "")
    configure_cache(getattr(args, "cache_dir", None))
    return runtime


def generation_config(args, mode: str, kb_path, selector: Optional[FewShotSelector] = None) -> dict:
    cfg = {
        "mode": mode,
        "n_examples": args.n_examples if mode == "with_kb" else 0,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "extra_body": parse_extra_body(getattr(args, "extra_body", None)),
        "kb": kb_info(kb_path) if mode == "with_kb" else None,
        "embed_model": args.embed_model if mode == "with_kb" else None,
    }
    if mode == "with_kb" and selector is not None:
        cfg["fewshot"] = selector.describe()
    return cfg


# Các trường cấu hình phải giống nhau thì mới được --resume nối tiếp một file samples.
RESUME_KEYS = ("mode", "n_examples", "prompt_style", "temperature", "seed", "max_tokens", "extra_body")
RESUME_FEWSHOT_KEYS = ("example_selection", "fewshot_format", "fewshot_asserts", "min_similarity", "mmr_lambda",
                       "retrieval_query")


def check_resume_compatible(samples_path: Path, runtime: RuntimeConfig, config: dict, force: bool = False) -> None:
    """Chặn --resume khi file samples cũ được sinh với cấu hình khác (trộn hai giao thức trong một file)."""
    if not samples_path.exists() or force:
        return
    meta_path = sidecar(samples_path, "meta.json")
    if not meta_path.exists():
        raise SystemExit(
            f"{samples_path} không có {meta_path.name} (có thể sinh bằng bản cũ / giao thức khác). "
            "Không --resume lên file này: dùng --run_tag mới hoặc --overwrite (thêm --force_resume nếu chắc chắn)."
        )
    latest = json.loads(meta_path.read_text(encoding="utf-8")).get("latest", {})
    old_cfg = latest.get("config", {})
    diffs = []
    if (latest.get("runtime") or {}).get("model") not in (None, runtime.model):
        diffs.append(f"model: {latest['runtime']['model']} -> {runtime.model}")
    for k in RESUME_KEYS:
        if k in old_cfg and old_cfg.get(k) != config.get(k):
            diffs.append(f"{k}: {old_cfg.get(k)} -> {config.get(k)}")
    old_fs, new_fs = old_cfg.get("fewshot") or {}, config.get("fewshot") or {}
    for k in RESUME_FEWSHOT_KEYS:
        if k in old_fs and old_fs.get(k) != new_fs.get(k):
            diffs.append(f"fewshot.{k}: {old_fs.get(k)} -> {new_fs.get(k)}")
    old_kb, new_kb = (old_cfg.get("kb") or {}).get("sha256"), (config.get("kb") or {}).get("sha256")
    if old_kb and new_kb and old_kb != new_kb:
        diffs.append("knowledge base đã thay đổi (sha256 khác)")
    if diffs:
        raise SystemExit(
            f"Không --resume được {samples_path.name}: cấu hình khác lần sinh trước:\n  - " + "\n  - ".join(diffs)
            + "\nDùng --run_tag mới, --overwrite, hoặc --force_resume nếu chắc chắn."
        )


def write_meta(samples_path: Path, runtime: RuntimeConfig, config: dict, stats: dict, started_at: str) -> dict:
    meta_path = sidecar(samples_path, "meta.json")
    history = []
    if meta_path.exists():
        try:
            history = json.loads(meta_path.read_text(encoding="utf-8")).get("invocations", [])
        except Exception:
            history = []
    entry = {
        "started_at": started_at,
        "finished_at": now_iso(),
        "command": sanitized_argv(),
        "runtime": describe_runtime(runtime),
        "config": config,
        "stats": stats,
        "environment": environment_info(),
    }
    meta = {"samples_file": str(samples_path), "latest": entry, "invocations": history + [entry]}
    write_json(meta_path, meta)
    return meta


def output_path_for(args, mode: str, prefix: str = "", results_dir=None) -> Path:
    if getattr(args, "output", None):
        return Path(args.output)
    base = Path(results_dir or args.results_dir) / model_slug(args.model)
    variant = fewshot_variant(args) if mode == "with_kb" else ""
    return base / f"{run_name(mode, args.n_examples, args.run_tag, prefix, variant)}_samples.jsonl"


# ───────────────────────────── các lệnh con ─────────────────────────────

def cmd_generate(args) -> int:
    runtime = runtime_from_args(args)
    if args.mode == "with_kb" and args.n_examples < 1:
        raise SystemExit("--mode with_kb cần --n_examples >= 1")
    kb_path = Path(args.kb_path or DEFAULT_KB_PATH)
    all_tasks = load_mbpp_plus_tasks()
    selector = build_selector(args, all_tasks, kb_path) if args.mode == "with_kb" else None
    tasks = all_tasks
    if args.limit:
        tasks = tasks[: args.limit]
        logger.warning("--limit %s: file samples sẽ thiếu bài, chỉ dùng để thử nghiệm", args.limit)
    out = output_path_for(args, args.mode)
    config = generation_config(args, args.mode, kb_path, selector)
    if args.resume:
        check_resume_compatible(out, runtime, config, args.force_resume)
    started = now_iso()
    stats = generate_all(
        runtime, tasks, out, selector, args.prompt_style, args.temperature, args.seed,
        args.max_tokens, args.max_retries, resume=args.resume, overwrite=args.overwrite, workers=args.workers,
            max_consecutive_errors=args.max_consecutive_errors,
    )
    stats["llm_calls"] = call_stats()
    write_meta(out, runtime, config, stats, started)
    print(format_check_report(check_samples(out)))
    return 0 if stats["errors"] == 0 else 3


def cmd_check(args) -> int:
    rep = check_samples(args.samples_file)
    print(format_check_report(rep))
    return 0 if rep["complete"] else 2


def cmd_evaluate(args) -> int:
    try:
        metrics = evaluate_samples(args.samples_file, args.evaluator, args.allow_incomplete, args.rerun, args.parallel)
    except IncompleteSamplesError as exc:
        print(exc)
        return 2
    print(json.dumps({k: metrics[k] for k in ("n_tasks", "base_pass@1", "plus_pass@1", "partial")}, ensure_ascii=False))
    return 0


def cmd_compare(args) -> int:
    cmp = compare_runs(args.a, args.b)
    print(format_comparison(cmp, args.a_label, args.b_label))
    if args.output:
        write_json(args.output, cmp)
    return 0


def cmd_run(args) -> int:
    runtime = runtime_from_args(args)
    if args.n_examples < 1:
        raise SystemExit("run cần --n_examples >= 1")
    kb_path = Path(args.kb_path or DEFAULT_KB_PATH)
    tasks = load_mbpp_plus_tasks()
    selector = build_selector(args, tasks, kb_path)
    if args.limit:
        tasks = tasks[: args.limit]
        logger.warning("--limit %s: kết quả là PARTIAL, không dùng để báo cáo", args.limit)
    summary = {
        "created_at": now_iso(),
        "command": sanitized_argv(),
        "runtime": describe_runtime(runtime),
        "environment": environment_info(),
        "runs": {},
    }
    paths = {}
    for mode, sel in (("baseline", None), ("with_kb", selector)):
        out = output_path_for(args, mode)
        paths[mode] = out
        config = generation_config(args, mode, kb_path, sel)
        if args.resume:
            check_resume_compatible(out, runtime, config, args.force_resume)
        started = now_iso()
        stats = generate_all(
            runtime, tasks, out, sel, args.prompt_style, args.temperature, args.seed,
            args.max_tokens, args.max_retries, resume=args.resume, overwrite=args.overwrite, workers=args.workers,
            max_consecutive_errors=args.max_consecutive_errors,
        )
        write_meta(out, runtime, config, stats, started)
        entry = {"samples_file": str(out), "config": config, "generation": stats}
        if not args.skip_eval:
            try:
                entry["metrics"] = evaluate_samples(out, args.evaluator, bool(args.limit), args.rerun, args.parallel)
            except IncompleteSamplesError as exc:
                entry["metrics"] = None
                entry["evaluation_error"] = str(exc)
                logger.error("%s", exc)
        summary["runs"][out.name.replace("_samples.jsonl", "")] = entry
    if not args.skip_eval and all(e.get("metrics") for e in summary["runs"].values()):
        cmp = compare_runs(paths["baseline"], paths["with_kb"])
        summary["comparison_baseline_vs_with_kb"] = cmp
        print(format_comparison(cmp, "baseline", f"KB n={args.n_examples}"))
    summary["llm_calls"] = call_stats()
    summary["finished_at"] = now_iso()
    variant = fewshot_variant(args)
    suffix = "_".join(x for x in (f"n{args.n_examples}{variant}", args.run_tag) if x)
    summary_path = Path(args.results_dir) / model_slug(args.model) / f"benchmark_summary_{suffix}.json"
    write_json(summary_path, summary)
    logger.info("Đã ghi %s", summary_path)
    return 0


def cmd_dev(args) -> int:
    from agent0.execution.strict_check import run_tests_isolated

    runtime = runtime_from_args(args)
    tasks = load_dev_tasks(args.data_file, args.dev_size, args.dev_seed)
    if args.limit:
        tasks = tasks[: args.limit]
    n_values = sorted(set(args.n_values))
    selectors: Dict[int, FewShotSelector] = {}
    if any(n > 0 for n in n_values):
        # Loại khỏi KB các mục sinh từ chính các bài dev (KB seed từ MBPP có thể chứa chúng).
        base = build_selector(argparse.Namespace(**{**vars(args), "n_examples": max(n_values)}), tasks,
                              Path(args.kb_path or DEFAULT_KB_PATH))
        for n in n_values:
            if n > 0:
                selectors[n] = FewShotSelector(
                    base.retriever, n, args.example_selection, args.fewshot_format, args.fewshot_asserts,
                    args.min_similarity, args.mmr_lambda, args.retrieval_query, args.seed,
                )
    logger.info("Tập dev: %s bài MBPP ngoài MBPP+ (dev_seed=%s)", len(tasks), args.dev_seed)

    by_id = {t["task_id"]: t for t in tasks}
    verdicts: Dict[int, Dict[str, Tuple[float, float]]] = {}
    summary = {
        "created_at": now_iso(), "command": sanitized_argv(), "runtime": describe_runtime(runtime),
        "dev": {"size": len(tasks), "dev_seed": args.dev_seed, "source": str(args.data_file)},
        "prompt_style": args.prompt_style, "temperature": args.temperature, "seed": args.seed,
        "kb": kb_info(args.kb_path or DEFAULT_KB_PATH) if selectors else None,
        "fewshot": next(iter(selectors.values())).describe() if selectors else None,
        "results": {},
    }
    for n in n_values:
        mode = "baseline" if n == 0 else "with_kb"
        args_n = argparse.Namespace(**{**vars(args), "n_examples": n, "output": None})
        out = output_path_for(args_n, mode, prefix="dev_", results_dir=args.results_dir)
        sel = selectors.get(n)
        config = generation_config(args_n, mode, args.kb_path or DEFAULT_KB_PATH, sel)
        if args.resume:
            check_resume_compatible(out, runtime, config, args.force_resume)
        started = now_iso()
        stats = generate_all(
            runtime, tasks, out, sel, args.prompt_style, args.temperature, args.seed,
            args.max_tokens, args.max_retries, resume=args.resume, overwrite=args.overwrite, workers=args.workers,
            max_consecutive_errors=args.max_consecutive_errors,
        )
        write_meta(out, runtime, config, stats, started)
        v = {}
        for row in read_jsonl(out):
            task = by_id.get(row["task_id"])
            if task is None:
                continue
            res = run_tests_isolated(row["solution"], task["test_list"], task["test_setup_code"], timeout=args.timeout)
            v[row["task_id"]] = (1.0 if res["passed"] else 0.0, 1.0 if res["passed"] else 0.0)
        verdicts[n] = v
        passed = sum(1 for t in by_id if v.get(t, (0.0, 0.0))[0] >= 0.5)
        summary["results"][f"n{n}"] = {
            "samples_file": str(out), "generation": stats,
            "pass_rate": round(100 * passed / max(1, len(by_id)), 2), "n_tasks": len(by_id),
        }
        logger.info("dev n=%s: %.2f%% (%s/%s, lỗi sinh %s)", n, 100 * passed / max(1, len(by_id)), passed, len(by_id), stats["errors"])

    ref = n_values[0]
    print(f"\nTập dev ({len(by_id)} bài MBPP ngoài MBPP+), so với n={ref}:")
    best_n, best_rate = ref, summary["results"][f"n{ref}"]["pass_rate"]
    for n in n_values:
        rate = summary["results"][f"n{n}"]["pass_rate"]
        line = f"  n={n}: {rate:.2f}%"
        if n != ref:
            common = sorted(t for t in by_id if t in verdicts[ref] and t in verdicts[n])
            cmp = compare_verdicts(verdicts[ref], verdicts[n], common)
            summary["results"][f"n{n}"]["vs_ref"] = cmp
            line += f"  (Δ {cmp['base']['delta_pp']:+.2f}, lật {cmp['base']['a_fail_b_pass']}/{cmp['base']['a_pass_b_fail']}, p = {cmp['base']['mcnemar_p']:.3f})"
            if rate > best_rate:
                best_n, best_rate = n, rate
        print(line)
    summary["best_n_by_pass_rate"] = best_n
    summary["llm_calls"] = call_stats()
    sig = [n for n in n_values if n != ref and summary["results"][f"n{n}"]["vs_ref"]["base"]["mcnemar_p"] < 0.05
           and summary["results"][f"n{n}"]["vs_ref"]["base"]["delta_pp"] > 0]
    summary["significantly_better_than_ref"] = sig
    print(f"=> n tốt nhất theo pass rate: {best_n}. "
          + (f"Tốt hơn n={ref} có ý nghĩa (p < 0.05): {sig}" if sig else f"Không n nào tốt hơn n={ref} có ý nghĩa thống kê."))
    variant = fewshot_variant(args)
    suffix = "_".join(x for x in (variant.lstrip("_"), args.run_tag) if x)
    summary_path = Path(args.results_dir) / model_slug(args.model) / (
        f"dev_summary_{suffix}.json" if suffix else "dev_summary.json"
    )
    write_json(summary_path, summary)
    logger.info("Đã ghi %s", summary_path)
    return 0


# ───────────────────────────── CLI ─────────────────────────────

def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", default=None, help="Mặc định lấy từ .env (GROQ_API_KEY / OLLAMA_API_KEY / OPENAI_API_KEY)")
    p.add_argument("--max_retries", type=int, default=None, help="Số lần thử lại khi lỗi tạm thời (mặc định 6)")
    p.add_argument("--workers", type=int, default=1, help="Số lời gọi model song song (vd 4–8; coi chừng giới hạn tốc độ của API)")
    p.add_argument("--api_key_env", default=None,
                   help="Đọc API key từ biến môi trường này (vd GEMINI_API_KEY cho --provider custom); không lộ key trong lệnh")
    p.add_argument("--extra_body", default=None,
                   help='JSON gộp vào body mỗi request, vd \'{"reasoning_effort": "low"}\' (gpt-oss trên Groq) hoặc '
                        '\'{"options": {"num_ctx": 8192}}\' (Ollama). Giá trị null xoá khoá.')
    p.add_argument("--max_consecutive_errors", type=int, default=10,
                   help="Dừng một lần sinh khi chừng này bài liên tiếp lỗi (0 = không dừng); hết quota thì dừng ngay")
    p.add_argument("--cache_dir", default=None,
                   help="Cache phản hồi LLM trên đĩa: chạy lại cùng cấu hình (temperature 0 + seed) không tốn lời gọi")


def _add_generation_args(p: argparse.ArgumentParser, with_mode: bool = True) -> None:
    if with_mode:
        p.add_argument("--mode", choices=["baseline", "with_kb"], default="baseline")
    p.add_argument("--n_examples", type=int, default=3, help="Số ví dụ few-shot lấy từ KB")
    p.add_argument("--kb_path", default=None, help=f"Mặc định: {DEFAULT_KB_PATH}")
    p.add_argument("--embed_model", default="mxbai-embed-large")
    p.add_argument("--embed_base_url", default="http://localhost:11434")
    p.add_argument("--example_selection", choices=FewShotSelector.SELECTIONS, default="retrieved",
                   help="retrieved: giống nhất theo embedding; random: ngẫu nhiên mỗi bài; fixed: một bộ cố định cho mọi bài")
    p.add_argument("--fewshot_format", choices=("tests", "full", "code"), default="tests",
                   help="tests: đề + assert ví dụ + lời giải (như prompt MBPP+); full: đề + reasoning + lời giải (bản cũ); code: đề + lời giải")
    p.add_argument("--fewshot_asserts", type=int, default=1, help="Số assert ví dụ trong mỗi ví dụ (định dạng tests)")
    p.add_argument("--min_similarity", type=float, default=0.0,
                   help="Ngưỡng cosine τ; ví dụ dưới τ bị bỏ, bài không còn ví dụ chạy zero-shot")
    p.add_argument("--mmr_lambda", type=float, default=1.0, help="< 1 để chọn ví dụ đa dạng theo MMR (vd 0.7)")
    p.add_argument("--retrieval_query", choices=("description", "prompt"), default="description",
                   help="description: truy vấn bằng mô tả bài (bỏ dòng assert); prompt: cả prompt")
    p.add_argument("--exclude_test_repaired", action="store_true",
                   help="Không dùng các mục KB mà test đã bị sửa cho khớp lời giải")
    p.add_argument("--prompt_style", choices=PROMPT_STYLES, default="evalplus",
                   help="evalplus: chỉ prompt gốc (chuẩn EvalPlus); with_tests: giao thức cũ, đưa base test vào prompt")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=1800)
    p.add_argument("--limit", type=int, default=None, help="Chỉ chạy N bài đầu (thử nghiệm; kết quả partial)")
    p.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--run_tag", default=None, help="Hậu tố tên lần chạy, vd seed1 -> baseline_seed1_samples.jsonl")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="Giữ các bài đã có, chỉ sinh bài còn thiếu")
    g.add_argument("--overwrite", action="store_true", help="Ghi đè file samples đã có")
    p.add_argument("--force_resume", action="store_true",
                   help="Cho --resume nối tiếp cả khi cấu hình khác lần sinh trước (không khuyến khích)")


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--evaluator", choices=["auto", "local", "docker"], default="auto",
                   help="auto: Docker trên macOS (nếu có), còn lại chạy trực tiếp")
    p.add_argument("--rerun", action="store_true", help="Chấm lại dù đã có *_eval_results.json")
    p.add_argument("--parallel", type=int, default=None)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MBPP+ (EvalPlus) cho Agent0 code KB")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log_file", default=None)
    sub = parser.add_subparsers(dest="action", required=True)

    gen = sub.add_parser("generate", help="Sinh lời giải cho MBPP+")
    _add_runtime_args(gen)
    _add_generation_args(gen)
    gen.add_argument("--output", default=None, help="Mặc định: <results_dir>/<model>/<run>_samples.jsonl")

    chk = sub.add_parser("check", help="Kiểm tra file samples đủ bài, không stub")
    chk.add_argument("--samples_file", required=True)

    ev = sub.add_parser("evaluate", help="Chấm file samples bằng EvalPlus")
    ev.add_argument("--samples_file", required=True)
    ev.add_argument("--allow_incomplete", action="store_true")
    _add_eval_args(ev)

    run = sub.add_parser("run", help="baseline + with_kb -> evaluate -> so sánh")
    _add_runtime_args(run)
    _add_generation_args(run, with_mode=False)
    _add_eval_args(run)
    run.add_argument("--skip_eval", action="store_true")

    cmp = sub.add_parser("compare", help="So sánh cặp hai lần chạy (McNemar)")
    cmp.add_argument("--a", required=True, help="samples.jsonl hoặc *_eval_results.json (vd baseline)")
    cmp.add_argument("--b", required=True, help="samples.jsonl hoặc *_eval_results.json (vd with_kb)")
    cmp.add_argument("--a_label", default="A")
    cmp.add_argument("--b_label", default="B")
    cmp.add_argument("--output", default=None)

    dev = sub.add_parser("dev", help="Chọn n trên tập dev (MBPP ngoài MBPP+)")
    _add_runtime_args(dev)
    _add_generation_args(dev, with_mode=False)
    dev.set_defaults(results_dir=str(DEFAULT_DEV_RESULTS_DIR))
    dev.add_argument("--n_values", type=int, nargs="+", default=[0, 1, 2, 3])
    dev.add_argument("--dev_size", type=int, default=200, help="Số bài dev lấy mẫu (tối đa 596)")
    dev.add_argument("--dev_seed", type=int, default=0)
    dev.add_argument("--data_file", default=str(DEFAULT_DATA_FILE))
    dev.add_argument("--timeout", type=float, default=10.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    setup_logger(log_file=args.log_file, log_level=args.log_level)
    handlers = {
        "generate": cmd_generate, "check": cmd_check, "evaluate": cmd_evaluate,
        "run": cmd_run, "compare": cmd_compare, "dev": cmd_dev,
    }
    return handlers[args.action](args)


if __name__ == "__main__":
    sys.exit(main())

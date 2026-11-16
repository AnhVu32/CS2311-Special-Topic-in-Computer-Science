"""Tín hiệu độ khó kiểu Agent0, không cần RL.

Agent0 thưởng cho curriculum agent theo độ bất định của executor: executor giải k lần, p̂ = tỉ lệ lời giải
"đúng" (ở bản toán: trùng đáp án đa số). Ở đây đề code có test, nên p̂ = tỉ lệ lời giải lấy mẫu qua toàn bộ
test (đã kiểm chứng) của đề:

    p_hat = (số lời giải qua test) / k
    r_unc = 1 - 2 * |p_hat - 0.5|       # 1 khi p_hat = 0.5; 0 khi p_hat = 0 hoặc 1

Đề có p_hat trong [low, high] (mặc định [0.25, 0.75] như Agent0) là đề ở biên năng lực của model.
p_hat theo subtopic được ghi vào summary.json để lần sinh đề sau gửi lại cho planner (`--feedback_from`).

Đồng tiến hoá không huấn luyện (`run_curriculum --p_hat_kb`): ở Agent0, vòng t+1 đo độ bất định bằng executor đã
được RL cập nhật ở vòng t. Ở đây trọng số không đổi, "executor vòng t" là model + KB_t, nên p_hat được đo khi lời giải
lấy mẫu có kèm ví dụ truy hồi từ KB_t (giống hệt cách benchmark dùng KB). Đề mà KB đã "phủ" trở nên dễ và bị lọc,
nên biên năng lực dịch chuyển khi KB lớn dần — thay cho bước cập nhật trọng số của executor.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent0.core.config import RuntimeConfig
from agent0.execution.agent import _signature_hints, generate_independent_solution
from agent0.execution.harness import run_harness
from agent0.execution.sandbox import record_tests

logger = logging.getLogger("agent0")

DEFAULT_BAND = (0.25, 0.75)


def uncertainty_reward(p_hat: float) -> float:
    return 1.0 - 2.0 * abs(p_hat - 0.5)


def _task_description(prompt: str) -> str:
    """Mô tả đề (bỏ dấu docstring và các dòng assert) — câu truy vấn KB, giống run_benchmark.task_description."""
    text = (prompt or "").strip().strip('"').strip()
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("assert")).strip()


def kb_few_shot(retriever, record: dict, n: int, fewshot_format: str = "tests") -> Tuple[str, Dict[str, Any]]:
    """n ví dụ giống nhất trong KB cho đề này, định dạng như prompt benchmark. Trả về (văn bản, thông tin truy hồi)."""
    examples = retriever.query(_task_description(record.get("prompt", "")), n=n) if retriever is not None and n > 0 else []
    text = retriever.format_few_shot(examples, style=fewshot_format, n_asserts=1) if examples else ""
    info = {
        "kb": str(getattr(retriever, "kb_path", "")) or None,
        "n": len(examples),
        "uids": [e.get("uid") for e in examples],
        "similarities": [e.get("similarity") for e in examples],
    }
    return text, info


def estimate_pass_rate(
    runtime: RuntimeConfig,
    record: dict,
    reference_code: str,
    k: int,
    temperature: float = 0.8,
    timeout: float = 20.0,
    base_seed: int = 2000,
    n_examples: int = 1,
    few_shot: str = "",
) -> Dict[str, Any]:
    """Sinh k lời giải độc lập (thấy đề, chữ ký hàm và `n_examples` assert đầu — như prompt MBPP+),
    chạy trên toàn bộ test của đề, trả về p_hat và r_unc. Lời gọi LLM lỗi không tính vào k.

    few_shot khác rỗng: mỗi lời giải được sinh kèm các ví dụ này (p_hat của "model + KB")."""
    tests = record_tests(record)
    setup = record.get("test_setup_code", "") or ""
    signatures = _signature_hints(reference_code, tests)
    examples = tests[: max(0, n_examples)]
    passes = errors = 0
    for i in range(max(0, k)):
        try:
            candidate = generate_independent_solution(
                runtime, record, signatures, examples, seed=base_seed + i, temperature=temperature, few_shot=few_shot
            )
        except Exception as exc:  # noqa: BLE001 - lỗi gọi model không phải là "giải sai"
            errors += 1
            logger.warning("p_hat sample failed | task_id=%s i=%s error=%s", record.get("task_id"), i, exc)
            continue
        if run_harness(candidate, tests, setup, timeout=timeout).get("passed"):
            passes += 1
    n = max(0, k) - errors
    out: Dict[str, Any] = {
        "k": n, "passes": passes, "errors": errors, "temperature": temperature, "n_examples_shown": len(examples),
        "with_kb": bool(few_shot), "p_hat": None, "r_unc": None,
    }
    if n > 0:
        p_hat = passes / n
        out["p_hat"] = round(p_hat, 4)
        out["r_unc"] = round(uncertainty_reward(p_hat), 4)
    logger.info("p_hat | task_id=%s passes=%s/%s p_hat=%s", record.get("task_id"), passes, n, out["p_hat"])
    return out


def band_verdict(p_hat: Optional[float], band: Tuple[float, float] = DEFAULT_BAND) -> str:
    """'in_band' | 'too_easy' | 'too_hard' | 'unknown'."""
    if p_hat is None:
        return "unknown"
    low, high = band
    if p_hat > high:
        return "too_easy"
    if p_hat < low:
        return "too_hard"
    return "in_band"


def summarize_by_group(rows: Iterable[dict], band: Tuple[float, float] = DEFAULT_BAND) -> List[dict]:
    """rows: {'group', 'difficulty', 'p_hat', 'r_unc'} -> thống kê p_hat theo subtopic/taxonomy."""
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("p_hat") is None:
            continue
        groups[(str(r.get("group") or "general"), str(r.get("difficulty") or "?"))].append(r)
    out = []
    for (group, difficulty), items in sorted(groups.items()):
        ps = [x["p_hat"] for x in items]
        out.append({
            "subtopic": group,
            "difficulty": difficulty,
            "n": len(items),
            "p_hat_mean": round(sum(ps) / len(ps), 3),
            "r_unc_mean": round(sum(uncertainty_reward(p) for p in ps) / len(ps), 3),
            "too_easy": sum(1 for p in ps if band_verdict(p, band) == "too_easy"),
            "too_hard": sum(1 for p in ps if band_verdict(p, band) == "too_hard"),
        })
    return out


def histogram(values: Iterable[Optional[float]], k: int) -> Dict[str, int]:
    """Đếm số đề theo từng giá trị p_hat (0/k, 1/k, ..., k/k)."""
    counts: Dict[str, int] = defaultdict(int)
    for v in values:
        if v is not None:
            counts[f"{v:.2f}"] += 1
    return dict(sorted(counts.items()))


def load_feedback(summary_path) -> List[dict]:
    """Đọc `uncertainty_by_subtopic` từ summary.json của lần chạy trước (để gửi cho planner)."""
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    fb = data.get("uncertainty_by_subtopic") or (data.get("stats") or {}).get("uncertainty_by_subtopic") or []
    if not fb:
        logger.warning("%s không có uncertainty_by_subtopic (lần chạy đó chưa bật --uncertainty_samples?)", summary_path)
    return fb
